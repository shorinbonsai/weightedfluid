import csv


a = zip(*csv.reader(open("/home/james/4F90/weightedfluid/data_07_15/raw.csv", "r")))
csv.writer(open("07_15transpose.csv",'w', newline='')).writerows(a)

