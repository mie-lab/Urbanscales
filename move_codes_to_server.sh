tar -czvf urbanscales.tar.gz urbanscales
echo "zipped urbanscales folder; moving to server"
scp urbanscales.tar.gz niskumar@XXX.XX.XXX.XXX:
scp config.py username@XXX.XX.XXX.XXX:WCS/config.py
ssh -t -t  niskumar@XXX.XX.XXX.XXX << EOF
  cd ~/WCS
  rm -rf urbanscales
  mv ~/urbanscales.tar.gz ./
  tar -xf urbanscales.tar.gz
  chmod +777 cache_osmnx/*
  echo "Logging out of server"
  exit
EOF
echo "Logged out of server"


