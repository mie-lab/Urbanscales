tar -czvf urbanscales.tar.gz urbanscales
echo "zipped urbanscales folder; moving to server"
scp urbanscales.tar.gz niskumar@kelut.sec.sg:
scp config.py niskumar@kelut.sec.sg:WCS/config.py
ssh -t -t  niskumar@kelut.sec.sg << EOF
  cd ~/WCS
  rm -rf urbanscales
  mv ~/urbanscales.tar.gz ./
  tar -xf urbanscales.tar.gz
  rm config.py 
  echo "Decompression complete and config.py deleted"
  echo "Logging out of server"
  exit
EOF
echo "Logged out of server"
scp config.py niskumar@kelut.sec.sg:WCS/config.py
echo "Config.py copied "

