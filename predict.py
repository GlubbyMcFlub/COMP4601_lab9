import os
import math
import numpy as np
import time

class RecommenderSystem:
    UNKNOWN_RATING = 0
    LIKED_RATING = 1
    PREDICTED_USER = "User1"

    def __init__(self, path):
        """
        Initialize the RecommenderSystem object.

        Parameters:
        - path (str): Path to the input data file.
        - neighbourhood_size (int): Size of the neighbourhood for recommendation.

        Returns:
        None
        """
        self.path = path
        self.num_users, self.num_items, self.users, self.items, self.ratings = self.read_data()

    def predictItems(self, predicted_user):
        """
        predicts item for predicted user

        Parameters:

        Returns:
        list of tuples containing an item that the predicted user may like
        and the number of paths leading to it
        """
        predicted_user_index = np.where(self.users == predicted_user)[0][0]
        # print(predicted_user_index)
        liked_items_of_predicted_user = np.where(self.ratings[predicted_user_index] == self.LIKED_RATING)[0]
        recommended_items = {}
        
        #for each liked item of predicted user
        for liked_item in liked_items_of_predicted_user:
            users_who_liked_item = np.where((self.ratings[:, liked_item] == self.LIKED_RATING) & (np.arange(self.num_users) != predicted_user_index))[0]
            # print(f"users who liked {self.items[liked_item]} = {self.users[users_who_liked_item]}")
            #for each user that liked that item
            for user in users_who_liked_item:
                items_liked_by_user = np.setdiff1d(np.where(self.ratings[user] == self.LIKED_RATING)[0], liked_items_of_predicted_user)
                # print(f"items liked by {self.users[user]} = {self.items[items_liked_by_user]}")
                #for each item that user liked
                for recommended_item in items_liked_by_user:
                    # print(f"incrementing {self.items[recommended_item]}")
                    #tally that item in list of tuples
                    if recommended_item not in recommended_items:
                        recommended_items[recommended_item] = 1
                    else:
                        recommended_items[recommended_item] += 1
        
        sorted_recommended_items = sorted(recommended_items.items(), key=lambda x: (x[1], x[0]), reverse=True)
        
        for item, path_count in sorted_recommended_items:
            print(f"{self.items[item]}: {path_count} paths")
        
        return sorted_recommended_items

    def read_data(self):
        """
        Read data from the input file and parse user-item ratings.

        Returns:
        Tuple containing:
        - num_users (int): Number of users in the dataset.
        - num_items (int): Number of items in the dataset.
        - users (numpy.ndarray): Array containing user labels.
        - items (numpy.ndarray): Array containing item labels.
        - ratings (numpy.ndarray): 2D array containing user-item ratings.
        """
        try:
            with open(self.path, 'r') as file:
                num_users, num_items = map(int, file.readline().split())
                users = np.array(file.readline().split())
                items = np.array(file.readline().split())
                ratings = np.array([[float(rating) if rating != self.UNKNOWN_RATING else self.UNKNOWN_RATING for rating in line.split()] for line in file])
                return num_users, num_items, users, items, ratings
        except ValueError as err:
            raise ValueError(f"Error: {str(err)}")
        except Exception as err:
            print(f"Error: {str(err)}")
            return None, None, None, None, None

def main():
    input_directory = "./input"
    files = [f for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]

    print("Select a file to process:")
    for index, file in enumerate(files):
        print(f"{index + 1}. {file}")

    try:
        selected_index = int(input("File to process: ")) - 1
        print(selected_index)
        selected_file = os.path.join(input_directory, files[selected_index])
        print(selected_file)

        recommender_system = RecommenderSystem(selected_file, )
        recommender_system.predictItems(recommender_system.PREDICTED_USER)
    except ValueError:
        print("Invalid input. Please enter a valid integer.")
        return

if __name__ == "__main__":
    main()
