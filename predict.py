import os
import math
import numpy as np
import time

class RecommenderSystem:
    MISSING_RATING = 0
    DEFAULT_NEIGHBOURHOOD_SIZE = 2

    def __init__(self, path, neighbourhood_size=DEFAULT_NEIGHBOURHOOD_SIZE):
        self.path = path
        self.neighbourhood_size = neighbourhood_size
        self.num_users, self.num_items, self.users, self.items, self.ratings = self.read_data()

    def read_data(self):
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
        num_common_users = len(common_users)

        if num_common_users == 0:
            return 0

        numerator = np.sum((self.ratings[common_users, item1_index] - average_ratings[common_users]) * (self.ratings[common_users, item2_index] - average_ratings[common_users]))
        item1_denominator = np.sum((self.ratings[common_users, item1_index] - average_ratings[common_users]) ** 2)
        item2_denominator = np.sum((self.ratings[common_users, item2_index] - average_ratings[common_users]) ** 2)

        denominator = math.sqrt(item1_denominator) * math.sqrt(item2_denominator)
        similarity = max(0, numerator / denominator) if denominator != 0 else 0

        return similarity

    def precompute_similarities(self):
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
        predicted_ratings = np.full((self.num_users, self.num_items), self.MISSING_RATING, dtype=float)

        for i in range(self.num_users):
            current_user_ratings = self.ratings[i]
            current_user_predicted_ratings = [0] * len(unrated_items)  # Initialize the list

            unrated_items = np.where(current_user_ratings == self.MISSING_RATING)[0]

            for j in unrated_items:
                neighbours = np.where((self.ratings[:, j] != self.MISSING_RATING) & (similarities[j] > 0))[0]
                adjusted_neighbourhood_size = min(self.neighbourhood_size, len(neighbours))

                if adjusted_neighbourhood_size == 0:
                    current_user_predicted_ratings[j] = np.nanmean(self.ratings[:, j])
                else:
                    top_neighbours = neighbours[np.argpartition(similarities[j, neighbours], -adjusted_neighbourhood_size)[-adjusted_neighbourhood_size:]]
                    sum_ratings = np.sum(similarities[j, top_neighbours] * self.ratings[:, j][top_neighbours])
                    total_similarity = np.sum(similarities[j, top_neighbours])
                    current_user_predicted_ratings[j] = max(0, min(5, sum_ratings / total_similarity)) if total_similarity != 0 else self.MISSING_RATING

            predicted_ratings[i] = current_user_predicted_ratings

        return predicted_ratings

    def fill_in_predicted_ratings(self):
        similarities = self.precompute_similarities()
        predicted_ratings = self.predict_ratings(similarities)
        expected_ratings = self.read_expected_output()

        print("\nOriginal Matrix:")
        self.print_matrix(self.ratings)

        self.ratings[np.isnan(self.ratings)] = predicted_ratings[np.isnan(self.ratings)]

        if expected_ratings is not None:
            self.compare_ratings(expected_ratings)
        else:
            print("No expected output available.")
            self.print_matrix(self.ratings)

    def predict_rating(self, userIndex, itemIndex, similarities):
        print(f"\nPredicting for user: {self.users[userIndex]}")
        print(f"Predicting for item: {self.items[itemIndex]}")

        neighbours = np.where((self.ratings[userIndex] != self.MISSING_RATING) & (similarities[itemIndex] > 0))[0]
        adjusted_neighbourhood_size = min(self.neighbourhood_size, len(neighbours))

        if adjusted_neighbourhood_size == 0:
            predict_rating = np.nanmean(self.ratings[userIndex])
            total_similarity = 0
            print("No valid neighbours found.")
        else:
            print(f"Found {adjusted_neighbourhood_size} valid neighbours:")
            neighbour_indices = neighbours[:adjusted_neighbourhood_size]
            neighbour_items = self.items[neighbour_indices]  # Use item indices instead of user indices
            similarities_values = similarities[itemIndex, neighbour_indices]

            for idx, (neighbour_item, similarity) in enumerate(zip(neighbour_items, similarities_values)):
                print(f"{idx + 1}. Item {neighbour_item} sim={similarity}")

            top_neighbours = neighbour_indices[np.argsort(similarities_values)[-adjusted_neighbourhood_size:]]
            sum_ratings = np.sum(similarities[itemIndex, top_neighbours] * self.ratings[userIndex, top_neighbours])
            total_similarity = np.sum(similarities[itemIndex, top_neighbours])

            print(f"Initial predicted value: {sum_ratings / total_similarity}")
            predict_rating = max(0, min(5, sum_ratings / total_similarity)) if total_similarity != 0 else np.nanmean(self.ratings[userIndex])

            print(f"Final predicted value: {predict_rating}")

        return predict_rating, total_similarity, adjusted_neighbourhood_size


    def find_mae(self):
        startTime = time.time()
        testsetSize = np.sum(~np.isnan(self.ratings))
        numerator = 0

        under_predictions = 0
        over_predictions = 0
        no_valid_neighbors = 0
        total_neighbors_used = 0

        for i in range(self.num_users):
            for j in range(self.num_items):
                if not np.isnan(self.ratings[i, j]):
                    temp = self.ratings[i, j]
                    self.ratings[i, j] = self.MISSING_RATING

                    similarities = self.precompute_similarities()
                    predicted_rating, total_similarity, adjusted_neighbourhood_size = self.predict_rating(i, j, similarities)

                    if not np.isnan(predicted_rating):
                        error = predicted_rating - temp
                        numerator += abs(error)

                        if error < 1:
                            under_predictions += 1
                        elif error > 5:
                            over_predictions += 1

                    if np.isnan(predicted_rating) or np.isinf(predicted_rating) or total_similarity == 0:
                        no_valid_neighbors += 1
                    else:
                        total_neighbors_used += adjusted_neighbourhood_size

                    self.ratings[i, j] = temp

        mae = numerator / testsetSize
        print(f"Numerator = {numerator}")
        print(f"TestsetSize = {testsetSize}")
        print(f"Total predictions: {testsetSize}")
        print(f"Total under predictions (< 1): {under_predictions}")
        print(f"Total over predictions (> 5): {over_predictions}")
        print(f"Number of cases with no valid neighbours: {no_valid_neighbors}")
        print(f"Average neighbors used: {total_neighbors_used / testsetSize}")
        print(f"MAE: {mae}")

        elapsedTime = time.time() - startTime
        if elapsedTime >= 60:
            minutes, seconds = divmod(elapsedTime, 60)
            print(f"Start: {int(minutes)}:{seconds:.3f} (m:ss.mmm)")
        else:
            print(f"Start: {elapsedTime:.3f}s")

        return mae

    def read_expected_output(self):
        expected_output_path = os.path.join("output", "out-" + os.path.basename(self.path))
        try:
            with open(expected_output_path, 'r') as file:
                expected_ratings = np.array([[float(rating) for rating in line.split()] for line in file.readlines()[3:]])
                return expected_ratings
        except FileNotFoundError:
            print(f"File not found: {expected_output_path}")
            return None

    def compare_ratings(self, expected_ratings):
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
        for row in matrix:
            print(" ".join(f"{rating:.2f}" if not np.isnan(rating) else f"{self.MISSING_RATING:.2f}" for rating in row))


def main():
    input_directory = "./input"
    files = [f for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]

    print("Select a file to process:")
    for index, file in enumerate(files):
        print(f"{index + 1}. {file}")

    try:
        selected_index = int(input("File to process: ")) - 1
        selected_file = os.path.join(input_directory, files[selected_index])
        recommender_system = RecommenderSystem(selected_file, neighbourhood_size=5)
        recommender_system.find_mae()
    except ValueError:
        print("Invalid input. Please enter a valid integer.")
        return

if __name__ == "__main__":
    main()
