import csv


a = zip(*csv.reader(open("/home/james/4F90/weightedfluid/modularityV3.csv", "r")))
csv.writer(open("modularityV3Transpose.csv",'w', newline='')).writerows(a)

