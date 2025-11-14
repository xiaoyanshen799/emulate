import concurrent.futures
import csv
import gc
import timeit
import traceback
from typing import Optional, Union

import psutil
from flwr.common import FitIns, FitRes, Parameters, Scalar
from flwr.server import Server, History
from flwr.server.client_proxy import ClientProxy
from flwr.server.server import FitResultsAndFailures, _handle_finished_future_after_fit

from .task import ZMQHandler


def fit_client(
        client: ClientProxy, ins: FitIns, timeout: Optional[float], group_id: int, logger
) -> tuple[ClientProxy, FitRes]:
    """Refine parameters on a single client."""
    client_round_start_time = timeit.default_timer()
    fit_res = None
    try:
        fit_res = client.fit(ins, timeout=timeout, group_id=group_id)
    except Exception as exc:
        tb_str = traceback.format_exc()
        logger.error("Failed to fit client %s to server. Traceback: %s", client.cid, tb_str)
    client_round_finish_time = timeit.default_timer()
    fit_res.metrics["client_round_start_time"] = client_round_start_time
    fit_res.metrics["client_round_finish_time"] = client_round_finish_time
    return client, fit_res


def fit_clients(
        client_instructions: list[tuple[ClientProxy, FitIns]],
        max_workers: Optional[int],
        timeout: Optional[float],
        group_id: int,
        logger
) -> FitResultsAndFailures:
    """Refine parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = set()
        for client_proxy, ins in client_instructions:
            submitted_fs.add(executor.submit(fit_client, client_proxy, ins, timeout, group_id, logger))

        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: list[tuple[ClientProxy, FitRes]] = []
    failures: list[Union[tuple[ClientProxy, FitRes], BaseException]] = []
    for future in finished_fs:
        _handle_finished_future_after_fit(
            future=future, results=results, failures=failures
        )
    return results, failures


class MyServer(Server):
    def __init__(self, *, client_manager, zmq_handler, log_path, strategy, logger):
        super().__init__(client_manager=client_manager, strategy=strategy)
        self.client_wise_file = open(f'{log_path}/fl_task_client_times.csv', 'w')
        self.overall_log_file = open(f'{log_path}/fl_task_overall_times.csv', 'w')
        self.client_wise_log = csv.writer(self.client_wise_file, dialect='unix')
        self.overall_log = csv.writer(self.overall_log_file, dialect='unix')
        self.zmq_handler = zmq_handler
        self.logger = logger

    def fit_round(self, server_round: int, timeout: Optional[float], ):
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )
        if not client_instructions:
            self.logger.info("configure_fit: no clients selected, cancel")
            return None
        self.logger.info(
            f"configure_fit: strategy sampled {len(client_instructions)} "
            f"clients (out of {self._client_manager.num_available()})",
        )

        if self.zmq_handler:
            round_clients = sorted([client_proxy.cid for client_proxy, _ in client_instructions])
            self.zmq_handler.send_data_to_server(ZMQHandler.MessageType.SERVER_TO_CLIENTS, round_clients)
            # time.sleep(0.1)

        self.logger.info(f"Memory usage before FitClients(): {psutil.Process().memory_info().rss / 1024 ** 2} MB")
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            group_id=server_round,
            logger=self.logger
        )
        self.logger.info(f"aggregate_fit: received {len(results)} results and {len(failures)} failures")
        gc.collect()
        # Aggregate training results
        aggregated_result: tuple[
            Optional[Parameters],
            dict[str, Scalar],
        ] = self.strategy.aggregate_fit(server_round, results, failures)

        parameters_aggregated, metrics_aggregated = aggregated_result
        return parameters_aggregated, metrics_aggregated, (results, failures)

    def fit(self, num_rounds: int, timeout: Optional[float]) -> tuple[History, float]:
        """Run federated averaging for a number of rounds."""
        history = History()

        # Initialize parameters
        self.logger.info("[INIT]")
        self.parameters = self._get_initial_parameters(server_round=0, timeout=timeout)
        self.logger.info("Starting evaluation of initial global parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            self.logger.info(f"initial parameters (loss, other metrics): {res[0]}, {res[1]}")
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])
        else:
            self.logger.info("Evaluation returned no results (`None`)")

        # Run federated learning for num_rounds

        start_time = timeit.default_timer()
        self.client_wise_log.writerow([
            'current_round',
            'client_id',
            'round_time',
            'server_send_model_time',
            'client_train_time',
            'client_send_model_time',
            'model_size_bytes',
        ])
        self.overall_log.writerow(['current_round', 'loss_cen', 'accuracy_cen', 'round_time', 'cumulative_time'])
        self.logger.info("Wrote to CSV files")

        for current_round in range(1, num_rounds + 1):
            round_start_time = timeit.default_timer()
            self.logger.info("")
            self.logger.info("[ROUND %s]", current_round)
            res_fit = self.fit_round(
                server_round=current_round,
                timeout=timeout,
            )
            round_end_time = timeit.default_timer()
            if res_fit is not None:
                parameters_prime, fit_metrics, (results, failures) = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime
                history.add_metrics_distributed_fit(
                    server_round=current_round, metrics=fit_metrics
                )
                self.log_csv_metrics(current_round, results)

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                round_time = round_end_time - round_start_time
                cumulative_time = timeit.default_timer() - start_time
                self.logger.info(f"fit progress: ({current_round}, {loss_cen}, {metrics_cen}, {round_time})")
                self.overall_log.writerow([current_round, loss_cen, metrics_cen['accuracy'], round_time, cumulative_time])
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )
                if metrics_cen['accuracy'] >= 0.79:
                    self.logger.info(f"Reaching Accuracy Level, Breaking!")
                    break

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed is not None:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed is not None:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )
            self.client_wise_file.flush()
            self.overall_log_file.flush()

        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        return history, elapsed

    def log_csv_metrics(self, current_round, results):
        for client_proxy, fit_res in results:
            metrics = fit_res.metrics
            client_id = metrics['client']
            server_to_client_time = metrics["computing_start_time"] - metrics["client_round_start_time"]
            computing_time = metrics["computing_finish_time"] - metrics["computing_start_time"]
            round_time = metrics["client_round_finish_time"] - metrics["client_round_start_time"]
            client_to_server_time = metrics["client_round_finish_time"] - metrics["computing_finish_time"]
            model_size_bytes = sum(len(tensor) for tensor in fit_res.parameters.tensors)

            metrics["server_send_model_time"] = server_to_client_time
            metrics["client_train_time"] = computing_time
            metrics["client_send_model_time"] = client_to_server_time
            metrics["model_size_bytes"] = model_size_bytes

            self.client_wise_log.writerow([
                current_round,
                client_id,
                round_time,
                server_to_client_time,
                computing_time,
                client_to_server_time,
                model_size_bytes,
            ])
