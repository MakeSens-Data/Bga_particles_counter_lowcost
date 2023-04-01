import numpy as np
from csv import reader
# Función principal
# open file in read mode
filename='data_in.csv'
output='data_out.csv'
with open(filename, 'r+') as f:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(f)
    coef0_pm10,coef1_pm10=1.8,3.9 
    coef0_pm25,coef1_pm25=2.3,4.6
    # Iterate over each row in the csv using reader object
    for row in csv_reader:
        # row variable is a list that represents a row in csv
        with open(output, 'a') as f2:
            if 'pm10_a' in row:
                f2.write('"'+str(row[0])+'"'+','+'"'+'pm10_ac'+'"'+','+str(float(row[2])*coef1_pm10+coef0_pm10)+'\n')
            if 'pm25_a' in row:
                f2.write('"'+str(row[0])+'"'+','+'"'+'pm25_ac'+'"'+','+str(float(row[2])*coef1_pm25+coef0_pm25)+'\n')
            else:
                f2.write('"'+str(row[0])+'"'+','+'"'+str(row[1])+'"'+','+'"'+str(row[2])+'"'+'\n')
