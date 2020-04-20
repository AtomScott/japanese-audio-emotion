# Run in ~/emotion
echo type hyphen seperated date ex. 2019-12-25 or 2020-01-01

read date

echo any parameters? leave blank and enter if none.

read params

python fujin.py -i JVAER -o docs/_posts -p $date-  $params