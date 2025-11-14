docker ps -q --filter "name=mn." | xargs docker stop
docker ps -q -a --filter "name=mn." | xargs docker rm