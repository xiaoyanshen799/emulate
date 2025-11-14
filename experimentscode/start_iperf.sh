#!/bin/bash

# Function to generate a random number from Gaussian distribution
# Uses Box-Muller transform
generate_gaussian_value() {
    mean=$1
    std_dev=$2

    # Generate two uniform random numbers between 0 and 1
    u1=$(echo "scale=10; $RANDOM/32767" | bc -l)
    u2=$(echo "scale=10; $RANDOM/32767" | bc -l)

    # Prevent log(0)
    while [ $(echo "$u1 < 0.0001" | bc -l) -eq 1 ]; do
        u1=$(echo "scale=10; $RANDOM/32767" | bc -l)
    done

    # Box-Muller transform to get normal distribution
    z0=$(echo "scale=10; sqrt(-2 * l($u1)) * c(2 * 3.14159265359 * $u2)" | bc -l)

    # Scale by std_dev and shift by mean
    value=$(echo "scale=0; $z0 * $std_dev + $mean" | bc -l)
    value=$(echo "if($value < 0) 0 else $value" | bc -l)
    # Truncate to non-negative values
    printf "%.0f" "$value"
}

# Function to start the server
start_server() {
    port=$1
    log_path=$2
    echo "Starting iperf server on port $port" >> "$log_path"/iperflogs.txt
    iperf3 -s -i 10 -p "$port" >> /dev/null 2>&1 &
    PID=$!

    sleep 0.2

    if ps -p $PID > /dev/null; then
        echo "Server started successfully with PID $PID..." >> "$log_path"/iperflogs.txt
        return 0
    else
        echo "Server failed to start .. Retrying ." >> "$log_path"/iperflogs.txt
        return 1
    fi
}

# Function to control dynamic traffic rate
control_traffic() {
    dst_ip=$1
    port=$2
    rate_mean=$3
    rate_std=$4
    time_mean=$5
    time_std=$6
    parallel=$7
    log_path=$8

    while true; do
        # Generate rate based on Gaussian distribution
        rate=$(generate_gaussian_value "$rate_mean" "$rate_std")

        # Ensure minimum rate (0.1 Mbps)
        rate=$(echo "if($rate < 0.1) 0.1 else $rate" | bc -l)

        # Generate time interval until next change based on Gaussian
        next_interval=$(generate_gaussian_value "$time_mean" "$time_std")

        # Ensure minimum interval (1 second)
        next_interval=$(echo "if($next_interval < 1) 1 else $next_interval" | bc -l)
        # Convert to integer for iperf
        next_interval=$(printf "%.0f" "$next_interval")

        echo "$(date): Setting traffic rate to $rate Mbps for approximately $next_interval seconds" >> "$log_path"/iperflogs.txt

        # Start iperf3 client with the current rate for the interval duration
        iperf3 -c "$dst_ip" -i 5 -t "$next_interval" -b "${rate}M" -p "$port" -P "$parallel" --connect-timeout 500 >> /dev/null 2>&1

        # Small delay to prevent potential race conditions
        sleep 0.5
    done
}

# Function to start the client with Gaussian distributed traffic
start_client() {
    dst_ip=$1
    port=$2
    rate_mean=$3     # Mean for traffic rate distribution
    rate_std=$4      # Standard deviation for traffic rate
    time_mean=$5     # Mean for time intervals distribution
    time_std=$6      # Standard deviation for time intervals
    parallel=$7
    log_path=$8

    echo "Starting iperf client to $dst_ip on port $port" >> "$log_path"/iperflogs.txt
    echo "- Rate follows Gaussian distribution with mean=$rate_mean, std_dev=$rate_std" >> "$log_path"/iperflogs.txt
    echo "- Rate change intervals follow Gaussian distribution with mean=$time_mean, std_dev=$time_std" >> "$log_path"/iperflogs.txt

    # Start the traffic controller process
    control_traffic "$dst_ip" "$port" "$rate_mean" "$rate_std" "$time_mean" "$time_std" "$parallel" "$log_path" &
    PID=$!

    sleep 0.2

    if ps -p $PID > /dev/null; then
        echo "Traffic controller started successfully with PID $PID..." >> "$log_path"/iperflogs.txt
        return 0
    else
        echo "Traffic controller failed to start .. Retrying ." >> "$log_path"/iperflogs.txt
        return 1
    fi
}

# Main script logic
if [[ "$1" == "server" ]]; then
    while ! start_server "$2" "$3"; do
        sleep 1  # Sleep for 1 second before retrying
    done

elif [[ "$1" == "client" ]]; then
    # New parameter structure: client port dst_ip rate_mean rate_std time_mean time_std log_path
    if [ $# -lt 8 ]; then
        echo "Missing parameters for client mode."
        echo "Usage: $0 client port dst_ip rate_mean rate_std time_mean time_std log_path"
        exit 1
    fi

    while ! start_client "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9"; do
        sleep 1  # Sleep for 1 second before retrying
    done

else
    echo "Invalid mode. Use 'server' or 'client'."
    echo "Server usage: $0 server port log_path"
    echo "Client usage: $0 client port dst_ip rate_mean rate_std time_mean time_std log_path"
    echo ""
    echo "  rate_mean: Mean value for the Gaussian distribution of traffic rate (in Mbps)"
    echo "  rate_std: Standard deviation for the Gaussian distribution of traffic rate"
    echo "  time_mean: Mean value for the Gaussian distribution of time between rate changes (in seconds)"
    echo "  time_std: Standard deviation for the Gaussian distribution of time intervals"
    exit 1
fi