import math
import numpy as np

#declare vars
path = "test.txt"
neighbourhoodSize = 2

#in case we need it
class unorderedDictionary:
    def __init__(self):
        self.data = {}

    def add(self, key1, key2, value):
        pair = frozenset([key1, key2])
        self.data[pair] = value

    def get(self, key1, key2):
        pair = frozenset([key1, key2])
        return self.data.get(pair, None)

# #read data
def readData(path):
    numUsers = None
    numItems = None
    users = []
    items = []
    ratings = []
    try:
        with open(path, 'r') as file:
            lineNum = 0
            for line in file:
                data = line.split()
                if lineNum == 0:
                    numUsers = int(data[0])
                    numItems = int(data[1])
                elif lineNum == 1:
                    for user in data:
                        users.append(user)
                elif lineNum == 2:
                    for item in data:
                        items.append(item)
                else:
                    ratings.append([int(rating) for rating in line.split()])
                lineNum += 1
    except Exception as err:
        print(f"error: {str(e)}")
    return numUsers, numItems, np.array(users), np.array(items), np.array(ratings)

#calculate similarities of all users
def predict(numUsers, numItems, users, items, ratings):
    # similarities = {}
    sameItemIndices = np.empty((numUsers, numUsers, numItems), dtype=np.object_)
    averageRatings = []
    # predictions = []

    # #get same items for each pair of users
    #optional??
    # for i in range(0, numUsers):
    #     for j in range(i + 1, numUsers):
    #         sameItems = []
    #         for z in range(0, numItems):
    #             if ratings[i][z] != -1 and ratings[j][z] != -1:
    #                 sameItems.append(z) #store the index
    #         sameItemsByPair[frozenset({users[i], users[j]})] = sameItems
    #         print(i, j, sameItemsByPair[frozenset({i, j})])
    print(sameItemIndices)
    for i in range(0, numUsers):
        for j in range(i + 1, numUsers):
            #get all elements not equal to -1
            user1Indices = np.where(ratings[i] != -1)[0]
            user2Indices = np.where(ratings[j] != -1)[0]

            #store same item indicies regardless of order
            intersectingIndices = np.intersect1d(user1Indices, user2Indices)
            # print(intersectingIndices)
            sameItemIndices[i, j] = intersectingIndices
            sameItemIndices[j, i] = intersectingIndices
            # print(sameItemIndices)
        # print(sameItemIndices)

    # print(sameItemIndices)
    
    # #get average ratings of each user
    averageRatings = np.ma.mean(np.ma.masked_equal(ratings, -1), axis=1)
    print(averageRatings)
    
    # # get Pearson's Correlation Coefficient (similarities) for each pair of users
    # for i in range(0, numUsers):
    #     for j in range(i + 1, numUsers):
    #         items = sameItemsByPair[frozenset({users[i], users[j]})]
    #         numerator = 0
    #         denominatorI = 0
    #         denominatorJ = 0
    #         for z in items:
    #             numerator += (ratings[i][z] - averageRatings[i]) * (ratings[j][z] - averageRatings[j])
    #             denominatorI += math.pow(ratings[i][z] - averageRatings[i], 2)
    #             denominatorJ += math.pow(ratings[j][z] - averageRatings[j], 2)
    #         denominatorI = math.sqrt(denominatorI)
    #         denominatorJ = math.sqrt(denominatorJ)
    #         similarities[frozenset({i, j})] = numerator/denominatorI/denominatorJ
    # print(similarities)
    # predict each unknown rating
    # for i in range(0, numUsers):
    #     neighbourhood = sorted(enumerate(similarities[i]), key=lambda x: x[1], reverse=True)
    #     neighbourhood = [index for index, _ in neighbourhood[:neighbourhoodSize]]
    #     print(neighbourhood)
        # for j in range(0, numItems):
            
        #     if ratings[i] != -1:
        #         continue
        #     #else
        #     for z in range(0, numUsers):
                
    return []

def main():
    numUsers, numItems, users, items, ratings = readData(path)
    # print(numUsers, numItems, users, items, ratings)
    dummy = predict(numUsers, numItems, users, items, ratings)
    # print(dummy)
    
    # dic = {}
    # x = frozenset({1,2,3})
    # dic[x] = 2
    # if frozenset({1,3,2}) in dic:
    #     print(dic[x])
    

    # print(numUsers, numItems, users, items, ratings)
    # print(similarities)

main()
