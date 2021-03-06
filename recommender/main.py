from model import load_data, calc_neighbours, print_top_songs_for_user
from metrics import rmse, mae, ndcg
from model import train_set, test_set


def main():
    # dataset has format like [user_id, song_id, play_count]
    file = 'train_triplets.txt'

    print ("Loading data...")
    load_data(file)

    print ("Starting evaluation...")
    calc_neighbours()
    print("Finished evaluations.")

    print_top_songs_for_user(1)

    print("Starting cross validation...")
    print("RMSE result: ", str(rmse(train_set, test_set)))
    print("MAE result: ", str(mae(train_set, test_set)))
    print("NDCG result: ", str(ndcg(train_set, test_set)))


main()

