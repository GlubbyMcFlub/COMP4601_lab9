import os
import math
import numpy as np
import time

class RecommenderSystem:
    MISSING_RATING = 0
    PREDICTED_USER = "User1"
    LIKED = 1
    UNKNOWN = 0

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
        print(predicted_user_index)
        #for each liked item of predicted user
        #for each user that liked that item
        #for each item that user liked
        #tally that item in list of tuples
        x = 0
        return x

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
                ratings = np.array([[float(rating) if rating != self.MISSING_RATING else self.MISSING_RATING for rating in line.split()] for line in file])
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
