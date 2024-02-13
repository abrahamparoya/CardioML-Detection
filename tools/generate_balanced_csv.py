import csv

def main():
	diseaseCounts = [0,0,0,0,0,0,0]
	writeString= [""]
	originalCsv = "/data/isip/data/tnmg_code/v1.0.0a/exams.csv"
	with open(originalCsv,"r") as inputFile:
		inputReader = csv.reader(inputFile)
		with open("../balanced_5000/balanced_5000.csv", "w") as outputFile:
			outputWriter = csv.writer(outputFile)
			writeString[0] = "exam_id"
			diseaseCategories = ["1dAVb", "RBBB", "LBBB", "SB", "ST", "AF"]
			writeString.extend(diseaseCategories)
			outputWriter.writerow(writeString)	
			writeString = [""]
			counter = 0	
			for row in inputReader:
				if(counter == 0):
					counter += 1
					continue

				diseaseString = row[4:10]
				if(diseaseString.count("True") == 0):
					# no disease present
					#
					diseaseCounts[0] += 1
					if(diseaseCounts[0] <= 714):
						# add row to balanced csv
						#
						writeString[0] = row[0]
						writeString.extend(diseaseString)
						outputWriter.writerow(writeString)
						writeString = [""]
				elif(diseaseString.count("True") == 1):
					# one disease present -> find out which one
					#
					presentDisease = diseaseString.index("True") + 1
					diseaseCounts[presentDisease] += 1
					if(diseaseCounts[presentDisease] <= 714):
						# add row to balanced csv
						#
						writeString[0] = row[0]
						writeString.extend(diseaseString)
						outputWriter.writerow(writeString)
						writeString = [""]
				if(diseaseCounts.count(714) == 7):
					break
if __name__ == "__main__":
	main()

