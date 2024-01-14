tar -czvf urbanscales.tar.gz urbanscales
echo "zipped urbanscales folder; moving to server"
scp urbanscales.tar.gz niskumar@172.25.185.141:
scp config.py niskumar@172.25.185.141:WCS/config.py
ssh -t -t  niskumar@172.25.185.141 << EOF
  cd ~/WCS
  rm -rf urbanscales
  mv ~/urbanscales.tar.gz ./
  tar -xf urbanscales.tar.gz
  chmod +777 cache_osmnx/*
  echo "Logging out of server"
  exit
EOF
echo "Logged out of server"


