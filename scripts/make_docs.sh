# Run in ~/emotion
echo type hyphen seperated date ex. 2019-12-25 or 2020-01-01

read date

python fujin.py -i code -o docs/_posts -p $date-  -s '' -x