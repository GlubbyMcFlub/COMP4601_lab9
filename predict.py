import os
import math
import numpy as np
import time

class RecommenderSystem:
    MISSING_RATING = 0.0
    DEFAULT_NEIGHBOURHOOD_SIZE = 2
    MIN_RATING = 1.0
    MAX_RATING = 5.0

    def __init__(self, path, neighbourhood_size=DEFAULT_NEIGHBOURHOOD_SIZE):
        """
        Initialize the RecommenderSystem object.

        Parameters:
        - path (str): Path to the input data file.
        - neighbourhood_size (int): Size of the neighbourhood for recommendation.

        Returns:
        None
        """
        self.path = path
        self.neighbourhood_size = neighbourhood_size
        self.num_users, self.num_items, self.users, self.items, self.ratings = self.read_data()

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

    def compute_similarity(self, item1_index, item2_index, common_users, average_ratings):
        """
        Compute the similarity between two items based on common user ratings.

        Parameters:
        - item1_index (int): Index of the first item.
        - item2_index (int): Index of the second item.
        - common_users (numpy.ndarray): Array of indices for users who have rated both items.
        - average_ratings (numpy.ndarray): Array of average ratings for each user.

        Returns:
        float: Similarity between the two items.
        """
        
        num_common_users = len(common_users)

        if num_common_users == 0:
            return 0

        numerator = np.sum((self.ratings[common_users, item1_index] - average_ratings[common_users]) * (self.ratings[common_users, item2_index] - average_ratings[common_users]))
        item1_denominator = np.sum((self.ratings[common_users, item1_index] - average_ratings[common_users]) ** 2)
        item2_denominator = np.sum((self.ratings[common_users, item2_index] - average_ratings[common_users]) ** 2)

        denominator = math.sqrt(item1_denominator * item2_denominator)
        similarity = max(0, numerator / denominator) if denominator != 0 else 0

        return similarity

    def precompute_similarities(self):
        """
        Precompute item similarities for the entire dataset.

        Returns:
        numpy.ndarray: 2D array containing precomputed item similarities.
        """
        
        similarities = np.zeros((self.num_items, self.num_items), dtype=np.float64)
        # calculate average ratings for all users, ignoring any missing ratings
        average_ratings = np.nanmean(np.where(self.ratings != self.MISSING_RATING, self.ratings, np.nan), axis=1)

        for i in range(self.num_items):
            for j in range(i + 1, self.num_items):
                # find common users who rated both items i and j
                item1_ratings = np.where(self.ratings[:, i] != self.MISSING_RATING)[0]
                item2_ratings = np.where(self.ratings[:, j] != self.MISSING_RATING)[0]
                common_users = np.intersect1d(item1_ratings, item2_ratings)
                # calculate ismilarity between items i and j based on common users
                similarity = self.compute_similarity(i, j, common_users, average_ratings)
                similarities[i, j] = similarity
                similarities[j, i] = similarity

        return similarities
    
    def sort_neighbours_by_similarity(self, neighbour_indices, similarities_values):
        """
        Sort neighbour indices and values based on similarity values in descending order.

        Parameters:
        - neighbour_indices (numpy.ndarray): Array of indices for neighbours.
        - similarities_values (numpy.ndarray): Array of similarity values.

        Returns:
        Tuple containing:
        - sorted_indices (numpy.ndarray): Sorted neighbour indices.
        - sorted_values (numpy.ndarray): Sorted similarity values.
        """
        sorted_indices = neighbour_indices[np.argsort(similarities_values)[::-1]]
        sorted_values = similarities_values[np.argsort(similarities_values)[::-1]]
        return sorted_indices, sorted_values

    def predict_rating(self, userIndex, itemIndex, similarities):
        """
        Predict a single rating for a user-item pair.

        Parameters:
        - userIndex (int): Index of the user.
        - itemIndex (int): Index of the item.
        - similarities (numpy.ndarray): 2D array containing precomputed item similarities.

        Returns:
        Tuple containing:
        - predict_rating (float): Predicted rating for the user-item pair.
        - total_similarity (float): Total similarity score of the neighbourhood.
        - adjusted_neighbourhood_size (int): Adjusted size of the neighbourhood used for prediction.
        """
        
        print(f"\nPredicting for user: {self.users[userIndex]}")
        print(f"Predicting for item: {self.items[itemIndex]}")

        # get all neighbours who rated the item, adjust neighbourhood size if necessary
        neighbour_indices = np.where((self.ratings[userIndex] != self.MISSING_RATING) & (similarities[itemIndex] > 0))[0]
        adjusted_neighbourhood_size = min(self.neighbourhood_size, len(neighbour_indices))

        # if no neighbours found, use average rating for item
        if adjusted_neighbourhood_size == 0:
            ratings_without_zeros = np.where(self.ratings[userIndex] != 0, self.ratings[userIndex], np.nan)
            predict_rating = np.nanmean(ratings_without_zeros)

            total_similarity = 0
            print("No valid neighbours found.")
        else:
            print(f"Found {adjusted_neighbourhood_size} valid neighbours:")
            sorted_indices, sorted_similarities = self.sort_neighbours_by_similarity(neighbour_indices, similarities[itemIndex, neighbour_indices])

            top_neighbours_indices = sorted_indices[:adjusted_neighbourhood_size]
            top_neighbours = self.items[top_neighbours_indices]

            for idx, (neighbour_item, similarity) in enumerate(zip(top_neighbours, sorted_similarities[:adjusted_neighbourhood_size])):
                print(f"{idx + 1}. Item {neighbour_item} sim={similarity}")

            sum_ratings = np.sum(similarities[itemIndex, top_neighbours_indices] * self.ratings[userIndex, top_neighbours_indices])
            total_similarity = np.sum(similarities[itemIndex, top_neighbours_indices])

            print(f"Initial predicted value: {sum_ratings / total_similarity}")
            predict_rating = max(0, min(5, sum_ratings / total_similarity)) if total_similarity != 0 else np.nanmean(self.ratings[userIndex])

            print(f"Final predicted value: {predict_rating}")

        return predict_rating, total_similarity, adjusted_neighbourhood_size

    def find_mae(self):
        """
        Find the Mean Absolute Error (MAE) for the prediction model.

        Returns:
        float: Mean Absolute Error (MAE) for the prediction model.
        """
        
        try:
            startTime = time.time()
            testsetSize = 0
            numerator = 0

            under_predictions = 0
            over_predictions = 0
            no_valid_neighbours = 0
            total_neighbours_used = 0

            for i in range(self.num_users):
                for j in range(self.num_items):
                    print(f"rating ({i}, {j}), = {self.ratings[i, j]}, bool = {not np.isnan(self.ratings[i, j]) and not self.ratings[i, j] == self.MISSING_RATING}")
                    if not np.isnan(self.ratings[i, j]) and not self.ratings[i, j] == self.MISSING_RATING:
                        testsetSize += 1
                        temp = self.ratings[i, j]
                        self.ratings[i, j] = self.MISSING_RATING

                        similarities = self.precompute_similarities()

                        # predict the rating for each user-item pair
                        predicted_rating, total_similarity, adjusted_neighbourhood_size = self.predict_rating(i, j, similarities)

                        if not np.isnan(predicted_rating):
                            error = abs(predicted_rating - temp)
                            numerator += error
                            if error < self.MIN_RATING:
                                print(f"predict: {predicted_rating}, temp: {temp}, error: {error}")

                            if error < self.MIN_RATING:
                                under_predictions += 1
                            elif error > self.MAX_RATING:
                                over_predictions += 1

                        if np.isnan(predicted_rating) or np.isinf(predicted_rating) or total_similarity == 0:
                            no_valid_neighbours += 1
                        else:
                            total_neighbours_used += adjusted_neighbourhood_size

                        self.ratings[i, j] = temp

            mae = numerator / testsetSize
            print(f"Numerator = {numerator}")
            print(f"TestsetSize = {testsetSize}")
            print(f"Total predictions: {testsetSize}")
            print(f"Total under predictions (< {1}): {under_predictions}")
            print(f"Total over predictions (> {5}): {over_predictions}")
            print(f"Number of cases with no valid neighbours: {no_valid_neighbours}")
            print(f"Average neighbours used: {total_neighbours_used / testsetSize}")
            print(f"MAE: {mae}")

            elapsedTime = time.time() - startTime
            if elapsedTime >= 60:
                minutes, seconds = divmod(elapsedTime, 60)
                print(f"Start: {int(minutes)}:{seconds:.3f} (m:ss.mmm)")
            else:
                print(f"Start: {elapsedTime:.3f}s")

            return mae
        except Exception as err:
            print(f"Error: {str(err)}")
            return None

def main():
    input_directory = "./input"
    files = [f for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]

    print("Select a file to process:")
    for index, file in enumerate(files):
        print(f"{index + 1}. {file}")

    try:
        selected_index = int(input("File to process: ")) - 1
        selected_file = os.path.join(input_directory, files[selected_index])
        recommender_system = RecommenderSystem(selected_file)
        recommender_system.find_mae()
    except ValueError:
        print("Invalid input. Please enter a valid integer.")
        return

if __name__ == "__main__":
    main()
