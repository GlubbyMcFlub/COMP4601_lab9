import os
import math
import numpy as np

class RecommenderSystem:
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
                        ratings.append([float(rating) if rating != '-1' else -1 for rating in data])
                    line_num += 1
        except ValueError:
            print("Error: Failed to parse data.")
            return None, None, None, None, None
        except Exception as err:
            print(f"Error: {str(err)}")
            return None, None, None, None, None

        return num_users, num_items, np.array(users), np.array(items), np.array(ratings)

    def compute_similarity(self, user1_ratings, user2_ratings, common_items):
        """
        Computes the Pearson correlation coefficient (PCC) between two users based on their common item ratings.

        Parameters:
        - user1_ratings (list): Ratings of user 1 for common items.
        - user2_ratings (list): Ratings of user 2 for common items.
        - common_items (list): List of indices for common items.

        Returns:
        - correlation (float): Pearson correlation coefficient.
        """

        num_common_items = len(common_items)

        if num_common_items == 0:
            return 0

        filt = [x for x in user1_ratings if x != -1]
        filt2 = [x for x in user2_ratings if x != -1]
        
        mean_user1 = sum(user1_ratings) / len(filt)
        mean_user2 = sum(user2_ratings) / len(filt2)

        numerator = sum((user1_ratings[i] - mean_user1) * (user2_ratings[i] - mean_user2) for i in range(num_common_items))
        denominator_user1 = math.sqrt(sum((user1_ratings[i] - mean_user1) ** 2 for i in range(num_common_items)))
        denominator_user2 = math.sqrt(sum((user2_ratings[i] - mean_user2) ** 2 for i in range(num_common_items)))
        print(f"numerator: {numerator:.2f}, denominator_user1: {denominator_user1:.2f}, denominator_user2: {denominator_user2:.2f}")

        if denominator_user1 * denominator_user2 == 0:
            return 0

        correlation = numerator / (denominator_user1 * denominator_user2)
        return correlation

    def precompute_similarities(self):
        """
        Precomputes the similarities between all pairs of users.

        Returns:
        - similarities (numpy.array): Matrix of precomputed similarities.
        """
        # TODO: only precompute similarity of ONE relation, not both
        
        similarities = np.zeros((self.num_users, self.num_users), dtype=np.float64)

        for i, user1 in enumerate(self.users):
            for j, user2 in enumerate(self.users):
                if i >= len(self.users) or j >= len(self.users):
                    print(f"Debug: i={i} or j={j} is out of bounds for self.users with size {len(self.users)}")
                    continue
                
                if self.users[i] == self.users[j]:
                    print(f"Skipping same user {user1} and {user2}")
                    continue

                user1_indices = np.where(self.ratings[i] != -1)[0]
                user2_indices = np.where(self.ratings[j] != -1)[0]

                intersecting_indices = np.intersect1d(user1_indices, user2_indices)
                common_items = intersecting_indices

                similarity = self.compute_similarity(self.ratings[i], self.ratings[j], common_items)
                print(f"Similarity between {user1} and {user2}: {similarity:.2f}")
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

        for i in range(self.num_users):
            current_user_ratings = self.ratings[i]
            current_user_predicted_ratings = np.full(self.num_items, -1, dtype=float)

            for j in range(self.num_items):
                if current_user_ratings[j] == -1:
                    neighbours = []

                    for k in range(self.num_users):
                        if i != k and self.ratings[k, j] != -1:
                            neighbours.append((k, similarities[i, k]))

                    neighbours.sort(key=lambda x: x[1], reverse=True)
                    top_neighbours = neighbours[:self.neighbourhood_size]
                    # TODO: Could also use a threshold for similarity

                    sum_ratings = 0
                    total_similarity = 0

                    for neighbour_index, similarity in top_neighbours:
                        neighbour_rating = self.ratings[neighbour_index, j]
                        # Calculate the deviation from the average
                        deviation = neighbour_rating - np.mean(self.ratings[neighbour_index])
                        print(f"Deviation from average for {self.users[neighbour_index]}: {deviation:.2f}")
                        
                        # Accumulate the weighted sum of deviations
                        sum_ratings += similarity * deviation
                        print(f"Weighted sum of deviations for {self.users[neighbour_index]}: {sum_ratings:.2f}")
                        total_similarity += abs(similarity)

                    if total_similarity > 0:
                        # Calculate predicted rating
                        print(f"Rating: {self.ratings[i]}")
                        user_average = np.mean(self.ratings[i][self.ratings[i] != -1])
                        influence = sum_ratings / total_similarity
                        predicted_rating = user_average + influence
                        print(f"Predicted rating for {self.users[i]} on {self.items[j]}: {predicted_rating:.2f}. User average: {user_average:.2f}, Influence: {influence:.2f}")
                        # Ensure rating is between 0 and 5
                        current_user_predicted_ratings[j] = max(0, predicted_rating)

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
                if self.ratings[i, j] == -1:
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
            print(f"Expected output file not found: {expected_output_path}")
            return None

    def compare_ratings(self, expected_ratings):
        """
        Helper function that compares the generated ratings with the expected ratings.

        Parameters:
        - expected_ratings (numpy.array): Matrix of expected ratings.
        """
        
        if expected_ratings is not None:
            if np.allclose(self.ratings, expected_ratings):
                print("Matrices are approximately equal.")
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
            print(" ".join(f"{rating:.2f}" if rating != -1 else "-1.00" for rating in row))

def main():
    input_directory = "./input"
    files = [f for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]

    print("Select a file to process:")
    for index, file in enumerate(files):
        print(f"{index + 1}. {file}")

    selected_index = int(input("File to process: ")) - 1
    selected_file = os.path.join(input_directory, files[selected_index])

    neighbourhood_size = 2

    recommender_system = RecommenderSystem(selected_file, neighbourhood_size)
    recommender_system.fill_in_predicted_ratings()

if __name__ == "__main__":
    main()
