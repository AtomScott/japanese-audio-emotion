fuser -n tcp -k 8888

xdg-open 'http://localhost:8888'
sshpass -f <(printf '%s\n' blingbling) ssh -N -L localhost:8888:localhost:8090 atom@130.158.126.185