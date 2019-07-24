echo "* Checking previous instance of API server and stopping ..."
for i in `ps -ef | grep search_endpoint | grep -v grep | awk '{ print $2 }'`; do kill -9 $i ; done
wait
dt=`date +%S%d`
path=$1
cd $path 
mkdir -p log
mv log/command.log log/command.log.$dt
echo "* Starting new API server instances"
nohup gunicorn --workers=3 --bind 0.0.0.0:5344 search_endpoint:app --access-logfile log/access.log --error-logfile log/error.log </dev/null >> log/command.log 2>&1 &
sleep 5
var=`ps -ef | grep search_endpoint | grep -v grep | awk '{ print $2 }'`
if [ -z "$var" ]
then
      echo "   * API server did not start correctly,"
      echo "   * run the this script as, bash /path/to/source/restartWrapper.sh /path/to/source "
      echo "   * or debug the API server, python search_endpoint.py" 
else
      echo "* API Server started successfully"
fi
echo $?

