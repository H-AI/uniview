MSG="
Please use provision as follows:
${yellow}./container/docker_build_uniview prod${end} :- builds docker in production mode
${yellow}./container/docker_build_uniview dev${end} :- build docker in developer mode
"

if [ "$1" == "prod" ]; then
    docker build -f container/Dockerfile -t uniview:latest . --build-arg ENV_MODE=prod
elif [ "$1" == "dev" ]; then
    docker build -f container/Dockerfile -t uniview:latest . --build-arg ENV_MODE=dev
else
    echo "$MSG"
fi