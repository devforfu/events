cls

# Launch configuration
$python_bin="python3"
$rawdir="Raw Data"
$linkdir="linked"
$statdir="statistics"
$events_names_file=$rawdir + "\AllEvents.txt"
$events_data_file=$rawdir + "\EURUSD_March.csv"
$stattmpl=$statdir + "\breakout_minchange_xx_pullback_yy.csv"
# test only
$linked=$linkdir + "\linked.csv"
#######################################

$events_names=[io.path]::combine($rawdir, $events_names_file)
#& $python_bin events.py -e $events_names_file -d $events_data_file -v --output-folder $linkdir
& $python_bin breakoutsbatch.py --max-change-low 10 --max-change-high 20 --min-pullback-low 10 --min-pullback-high 20 -i $linked -o $stattmpl -t
