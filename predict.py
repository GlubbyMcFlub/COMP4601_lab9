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
        try:
            with open(self.path, 'r') as file:
                num_users, num_items = map(int, file.readline().split())
                users = np.array(file.readline().split())
                items = np.array(file.readline().split())
                ratings = np.array([[float(rating) if rating != self.MISSING_RATING else self.MISSING_RATING for rating in line.split()] for line in file])
                return num_users, num_items, users, items, ratings
        except ValueError:
            print("Error: Failed to parse data.")
            return None, None, None, None, None
        except Exception as err:
            print(f"Error: {str(err)}")
            return None, None, None, None, None

    def compute_similarity(self, item1_index, item2_index, common_users, average_ratings):
        """
        Computes the adjusted cosine similarity between two items based on user ratings.

        Parameters:
        - item1_index (int): Index of the first item.
        - item2_index (int): Index of the second item.
        - common_users (numpy.array): Indices of common users who rated both items.
        - average_ratings (numpy.array): Average ratings for each user.

        Returns:
        - similarity (float): Cosine similarity.
        """
        num_common_users = len(common_users)

        if num_common_users == 0:
            return 0

        numerator = 0
        item1_denominator = 0
        item2_denominator = 0
        
        # Iterate of the common users to determine each component of the similarity formula
        for i in common_users:
            numerator += (self.ratings[i, item1_index] - average_ratings[i]) * (self.ratings[i, item2_index] - average_ratings[i])
            item1_denominator += (self.ratings[i, item1_index] - average_ratings[i]) ** 2
            item2_denominator += (self.ratings[i, item2_index] - average_ratings[i]) ** 2
            
        # Cosine similarity formula
        denominator = math.sqrt(item1_denominator) * math.sqrt(item2_denominator)
        similarity = numerator / denominator if denominator != 0 else 0

        return similarity

    def precompute_similarities(self):
        """
        Precomputes the adjusted cosine similarities between all pairs of items.

        Returns:
        - similarities (numpy.array): Matrix of precomputed item similarities.
        """
        similarities = np.zeros((self.num_items, self.num_items), dtype=np.float64)
        average_ratings = np.nanmean(np.where(self.ratings != self.MISSING_RATING, self.ratings, np.nan), axis=1)

        for i in range(self.num_items):
            for j in range(i + 1, self.num_items):
                item1_ratings = np.where(self.ratings[:, i] != self.MISSING_RATING)[0]
                item2_ratings = np.where(self.ratings[:, j] != self.MISSING_RATING)[0]
                common_users = np.intersect1d(item1_ratings, item2_ratings)
                similarity = self.compute_similarity(i, j, common_users, average_ratings)
                similarities[i, j] = similarity
                similarities[j, i] = similarity

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

        # TODO: implement similar strat of unrated_items but for users?
        for i in range(self.num_users):
            current_user_ratings = self.ratings[i]
            current_user_predicted_ratings = np.full(self.num_items, self.MISSING_RATING, dtype=float)
            unrated_items = np.where(current_user_ratings == self.MISSING_RATING)[0]

            for j in unrated_items:
                # Find neighbours who rated item j and have > 0 similarity
                neighbours = np.where((self.ratings[i] != self.MISSING_RATING) & (similarities[j] > 0))[0]
                adjusted_neighbourhood_size = min(self.neighbourhood_size, len(neighbours))
                # Find top neighbours with highest similarity for item j
                top_neighbours = np.argpartition(similarities[j, neighbours], -adjusted_neighbourhood_size)[-adjusted_neighbourhood_size:]
                top_neighbours = neighbours[top_neighbours]
                # Calculate predicted rating for item j
                sum_ratings = np.sum(similarities[j, top_neighbours] * self.ratings[i, top_neighbours])
                total_similarity = np.sum(similarities[j, top_neighbours])

                current_user_predicted_ratings[j] = max(0, min(5, sum_ratings / total_similarity)) if total_similarity != 0 else self.MISSING_RATING

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
                expected_ratings = np.array([[float(rating) for rating in line.split()] for line in file.readlines()[3:]])
                return expected_ratings
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

    try:
        selected_index = int(input("File to process: ")) - 1
        selected_file = os.path.join(input_directory, files[selected_index])
        recommender_system = RecommenderSystem(selected_file)
        recommender_system.fill_in_predicted_ratings()
    except ValueError:
        print("Invalid input. Please enter a valid integer.")
        return

if __name__ == "__main__":
    main()