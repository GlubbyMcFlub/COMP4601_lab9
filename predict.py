import os
import math
import numpy as np

class RecommenderSystem:
    MISSING_RATING = -1
    
    def __init__(self, path, neighbourhood_size=2):
        """
        Initialize the RecommenderSystem with the given file path and neighbourhood size.

        Parameters:
        - path (str): File path of the input data.
        - neighbourhood_size (int): Size of the neighbourhood for collaborative filtering.
        """
        
        self.path = path
        self.neighbourhood_size = neighbourhood_size
        self.num_users, self.num_items, self.users, self.items, self.ratings = self.read_data()

    def read_data(self):
        """
        Parses the text file and returns the number of users, number of items, users, items, and ratings.

        Returns:
        - num_users (int): Number of users.
        - num_items (int): Number of items.
        - users (numpy.array): Array of user names.
        - items (numpy.array): Array of item names.
        - ratings (numpy.array): Array of user-item ratings.
        """
        
        num_users = None
        num_items = None
        users = []
        items = []
        ratings = []

        try:
            with open(self.path, 'r') as file:
                line_num = 0
                for line in file:
                    data = line.split()
                    if line_num == 0:
                        num_users = int(data[0])
                        num_items = int(data[1])
                    elif line_num == 1:
                        users = data
                        
                        if len(users) != num_users:
                            raise ValueError("Number of users does not match.")
                    elif line_num == 2:
                        items = data

                        if len(items) != num_items:
                            raise ValueError("Number of items does not match.")
                    else:
                        ratings.append([float(rating) if rating != self.MISSING_RATING else self.MISSING_RATING for rating in data])
                    line_num += 1
        except ValueError:
            print("Error: Failed to parse data.")
            return None, None, None, None, None
        except Exception as err:
            print(f"Error: {str(err)}")
            return None, None, None, None, None

        return num_users, num_items, np.array(users), np.array(items), np.array(ratings)

    def compute_similarity(self, item1_ratings, item2_ratings, common_users):
        """
        Computes the Pearson correlation coefficient (PCC) between two users based on their common item ratings.

        Parameters:
        - item1_ratings (list): Ratings of user 1 for common users.
        - item2_ratings (list): Ratings of user 2 for common users.
        - common_items (list): List of indices for common items.

        Returns:
        - correlation (float): Pearson correlation coefficient.
        """
        num_common_items = len(common_users)

        if num_common_items == 0:
            return 0

        item1_ratings_common = [item1_ratings[i] for i in common_users]
        item2_ratings_common = [item2_ratings[i] for i in common_users]
        item1_ratings_all = [x for x in item1_ratings if x != self.MISSING_RATING]
        item2_ratings_all = [x for x in item2_ratings if x != self.MISSING_RATING]

        mean_item1 = np.mean(item1_ratings_all)
        mean_item2 = np.mean(item2_ratings_all)

        numerator = sum((item1_ratings_common[i] - mean_item1) * (item2_ratings_common[i] - mean_item2) for i in range(num_common_items))
        denominator_item1 = math.sqrt(sum((item1_ratings_common[i] - mean_item1) ** 2 for i in range(num_common_items)))
        denominator_item2 = math.sqrt(sum((item2_ratings_common[i] - mean_item2) ** 2 for i in range(num_common_items)))

        if denominator_item1 * denominator_item2 == 0:
            return 0

        correlation = numerator / (denominator_item1 * denominator_item2)
        return correlation


    def precompute_similarities(self):
        """
        Precomputes the similarities between all pairs of users.

        Returns:
        - similarities (numpy.array): Matrix of precomputed similarities.
        """
        
        similarities = np.zeros((self.num_items, self.num_items), dtype=np.float64)

        for i in range(self.num_items):
            # Calculate for unique pairs of items
            for j in range(i + 1, self.num_items): 
                item1_ratings = np.where(self.ratings[:, i] != self.MISSING_RATING)[0]
                item2_ratings = np.where(self.ratings[:, j] != self.MISSING_RATING)[0]

                intersecting_ratings = np.intersect1d(item1_ratings, item2_ratings)
                common_users = intersecting_ratings

                similarity = self.compute_similarity(self.ratings[:, i], self.ratings[:, j], common_users)
                similarities[i, j] = similarity
                similarities[j, i] = similarity
                print(f"item {i},{j} similarity = {similarity}")
        return similarities

    def predict_ratings(self, similarities):
        """
        Predicts the ratings for items that a user has not rated yet, using collaborative filtering.

        Parameters:
        - similarities (numpy.array): Matrix of precomputed similarities.

        Returns:
        - predicted_ratings (list): List of predicted ratings for each user.
        """
        
        predicted_ratings = []

        for i in range(self.num_users):
            current_user_ratings = self.ratings[i]
            current_user_predicted_ratings = np.full(self.num_items, self.MISSING_RATING, dtype=float)

            for j in range(self.num_items):
                if current_user_ratings[j] == self.MISSING_RATING:
                    neighbours = []

                    for k in range(self.num_users):
                        if i != k and self.ratings[k, j] != self.MISSING_RATING:
                            neighbours.append((k, similarities[i, k]))

                    neighbours.sort(key=lambda x: x[1], reverse=True)
                    top_neighbours = neighbours[:self.neighbourhood_size]
                    # TODO: Could also use a threshold for similarity
                    sum_ratings = 0
                    total_similarity = 0

                    for neighbour_index, similarity in top_neighbours:
                        neighbour_rating = self.ratings[neighbour_index, j]
                        # Calculate the deviation from the average
                        filtered_ratings = [x for x in self.ratings[neighbour_index] if x != self.MISSING_RATING]
                        deviation = neighbour_rating - np.mean(filtered_ratings)
                        # Accumulate the weighted sum of deviations
                        sum_ratings += similarity * deviation
                        total_similarity += similarity

                    # Calculate predicted rating
                    user_average = np.mean(self.ratings[i][self.ratings[i] != self.MISSING_RATING])
                    influence = sum_ratings / total_similarity
                    predicted_rating = user_average + influence
                    # Ensure rating is between 0 and 5
                    current_user_predicted_ratings[j] = max(0, min(5, predicted_rating))

            predicted_ratings.append(current_user_predicted_ratings)

        return predicted_ratings

    def fill_in_predicted_ratings(self):
        """
        Fills in the predicted ratings for all users and items.
        Compares the generated ratings with the expected ratings if available.
        """
        
        similarities = self.precompute_similarities()
        predicted_ratings = self.predict_ratings(similarities)
        expected_ratings = self.read_expected_output()
        
        print("\nOriginal Matrix:")
        self.print_matrix(self.ratings)  

        for i in range(self.num_users):
            for j in range(self.num_items):
                if self.ratings[i, j] == self.MISSING_RATING:
                    self.ratings[i, j] = predicted_ratings[i][j]

        if expected_ratings is not None:
            self.compare_ratings(expected_ratings)
        else:
            print("No expected output available.")
            self.print_matrix(self.ratings)

    def read_expected_output(self):
        """
        Helper function that parses the expected output file.

        Returns:
        - expected_ratings (numpy.array): Matrix of expected ratings.
        """
        
        expected_output_path = os.path.join("output", "out-" + os.path.basename(self.path))
        try:
            with open(expected_output_path, 'r') as file:
                expected_ratings = []
                next(file)
                next(file)
                next(file)
                for line in file:
                    expected_ratings.append([float(rating) for rating in line.split()])
                return np.array(expected_ratings)
        except FileNotFoundError:
            print(f"File not found: {expected_output_path}")
            return None

    def compare_ratings(self, expected_ratings):
        """
        Helper function that compares the generated ratings with the expected ratings.

        Parameters:
        - expected_ratings (numpy.array): Matrix of expected ratings.
        """
        
        if expected_ratings is not None:
            rounded_generated_ratings = np.round(self.ratings, decimals=2)
            rounded_expected_ratings = np.round(expected_ratings, decimals=2)

            if np.allclose(rounded_generated_ratings, rounded_expected_ratings):
                print("Matrices are equalish, good enough.")
            else:
                print("Matrices are NOT equal.") 

            print("\nGenerated Matrix:")
            self.print_matrix(self.ratings)

            print("\nExpected Matrix:")
            self.print_matrix(expected_ratings)
        else:
            print("Expected ratings are not available.")

    def print_matrix(self, matrix):
        """
        Helper function that prints a matrix with at most two decimal places.

        Parameters:
        - matrix (numpy.array): Matrix to be printed.
        """
        
        for row in matrix:
            print(" ".join(f"{rating:.2f}" if rating != self.MISSING_RATING else f"{self.MISSING_RATING:.2f}" for rating in row))


def main():
    input_directory = "./input"
    files = [f for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]

    print("Select a file to process:")
    for index, file in enumerate(files):
        print(f"{index + 1}. {file}")

    selected_index = int(input("File to process: ")) - 1
    selected_file = os.path.join(input_directory, files[selected_index])

    recommender_system = RecommenderSystem(selected_file)
    recommender_system.fill_in_predicted_ratings()

if __name__ == "__main__":
    main()
