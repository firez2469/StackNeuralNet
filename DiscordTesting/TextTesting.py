from SebNN import NeuralNet
import numpy as np

globalMax = 650
def convertToAscii(filename,person):
    with open(filename,encoding="utf8") as f:
        contents = f.readlines()
        endVals =[]
        for line in contents:
            splitLine = line.split('|')
            newLine = splitLine[1]
            lineAsciis = []

            for char in newLine:
                newVal = ord(char)/255
                lineAsciis.append(newVal)
            #ensure text is at least 100
            while len(lineAsciis)<globalMax:
                lineAsciis.append(0)

            #assign expected outputs if 0 or 1
            if splitLine[0]==person:
                lineAsciis.append(1)
            else:
                lineAsciis.append(0)
            endVals.append(lineAsciis)
        return endVals

def rawConvertToAscii(text):
    out = []
    for char in text:
        out.append(ord(char))
    while len(out)<globalMax:
        out.append(0)
    return out

print('Converting...')
asciiData = convertToAscii('./out','firez2469')
print('Data converted!')
n_inputs = len(asciiData[0]) - 1

n_outputs = len(set([row[-1] for row in asciiData]))

net = NeuralNet(n_inputs,3,n_outputs)

net.fit(asciiData,0.3,500)

out = net.run(asciiData)
print(out)

expected = []
for row in asciiData:
    expected.append(row[len(row)-1])
print(expected)
noWork = net.run([rawConvertToAscii("the capital and lowercase describing that the first one has a higher impact, being multiplied by 16")])
yesWork = net.run([rawConvertToAscii("What version we on again?")])
print("Flex was recognized as:",yesWork)
print("Other person was recognized as:",noWork)

#print(net.network)


#Basic Initial example
def runExample():
        dataset = [[1,0,0,0],
                   [0,0,1,0],
                   [0,1,0,0],
                   [1,1,0,1],
                   [1,1,1,1],
                   [0,0,0,0],
                   [1,0,1,1]]
        n_inputs = len(dataset[0])-1
        n_outputs = len(set([row[-1] for row in dataset]))
        #test_n_outputs = len(set([row[-1] for row in dataset]))

        nn = NeuralNet(n_inputs,2,n_outputs)
        nn.fit(dataset,0.5,100)
        #nn.predict()

        newDataSet = [[1,1,1],
                      [1,0,0],
                      [1,1,1]]

        out = nn.run(newDataSet)
        print(out)