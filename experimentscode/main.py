import os
import tomli

from mininet.log import setLogLevel, info
from mininet.net import Containernet
from mininet.cli import CLI
from mininet.link import TCLink
from mininet.node import OVSSwitch, RemoteController

from run_and_monitor_exp import ExperimentRunner
from topohub_bg_gen import BGTrafficGenerator
from topohub_topology import MyTopo2


class MyMininet(Containernet):
    def __init__(self, **kwargs):
        Containernet.__init__(self, **kwargs)

    def get_fl_hosts(self):
        fl_server = self.get("flserver")
        fl_clients = [host for host in self.hosts if "flclient" in host.name]
        return fl_server, fl_clients

    def get_bg_hosts(self):
        bg_clients = {host.name: host for host in self.hosts if "bgclient" in host.name}
        return bg_clients

    def __enter__(self):
        info('*** Start Network\n')
        self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            print(f"Exception type: {exc_type}")
            print(f"Exception value: {exc_value}")
            print(f"Traceback: {traceback}")
        else:
            print("Exiting the context without exception")
        info('*** Stopping network\n')
        self.stop()
        return True  # If False, the exception will be re-raised


def start():
    controller = RemoteController('c0', ip=sdn_conf["ip"], port=sdn_conf["port"]) if sdn_conf["enable"] else None
    topo_creator = MyTopo2(topo_conf=topo_conf, client_conf=client_conf)
    net = MyMininet(topo=topo_creator, switch=OVSSwitch, link=TCLink, controller=controller)

    (fl_server, fl_clients), bg_clients = net.get_fl_hosts(), net.get_bg_hosts()

    logs_path = f"{other_conf['logs-path']}/{other_conf['name']}"
    bg_gen = BGTrafficGenerator(bg_traffic_conf, bg_clients, topo_creator.nodes_data, topo_creator.links_data, logs_path)
    exp_runner = ExperimentRunner(fl_server=fl_server, fl_clients=fl_clients, onos_server=sdn_conf["ip"], logs_path=logs_path)

    with net:
        print("Starting: ", other_conf['name'])
        CLI(net)
        with exp_runner:
            if topo_conf["enable-bg-traffic"]:
                with bg_gen:
                    exp_runner.start_experiment()
            else:
                exp_runner.start_experiment()
    os.system("pkill -9 -f 'bazel|onos'")


if __name__ == '__main__':
    setLogLevel('info')

    with open('settings.toml', 'rb') as f:
        config = tomli.load(f)

    topo_conf = config['topology']
    bg_traffic_conf = config['bg-traffic']
    client_conf = config['docker-client']
    sdn_conf = config['sdn-controller']
    other_conf = config['other']

    start()
