# For use in hung up interactive DDP jobs
kill %1
sleep 5
kill %1
nuisance=$(ps aux | grep 'joy47' | grep 'python' | grep -v " Z " | grep -v grep | awk '{print $2}')
for i in $nuisance; do kill -9 $i; done