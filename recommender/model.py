import random
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

n = 10
users_count = 1000
songs_count = 384546  # count of unique songs

users = dict()
songs = dict()
nearest_neighbours = dict()

train_set = sparse.lil_matrix((users_count, songs_count))
test_set = sparse.lil_matrix((users_count, songs_count))


def get_id(dict, key):
    if key not in dict:
        dict[key] = len(dict)
    return dict[key]


def load_data(file):
    with open(file, 'r') as f:
        for line in f:
            user, song, play_count = line.split('\t')

            user_id = get_id(users, user)
            song_id = get_id(songs, song)

            if (user_id + 1) > users_count:
                break
            train_set[user_id, song_id] = play_count
            if random.randint(1, 10) > 8:
                test_set[user_id, song_id] = play_count


def get_n_neighbours(user):
    neighbours = []
    for neighbour in range(train_set.shape[0]):
        cos = cosine_similarity(train_set[user], train_set[neighbour])
        neighbours.append((cos, neighbour))
    return sorted(neighbours, reverse=True)[1:(n + 1)]


def calc_neighbours():
    for user in range(train_set.shape[0]):
        nearest_neighbours[user] = get_n_neighbours(user)


def get_rating(user, song):
    neighbours = nearest_neighbours[user]
    upper_sum = 0.0
    lower_sum = 0.0
    for (cos, neighbour) in neighbours:
        rate = train_set[neighbour, song]
        if rate != 0:
            upper_sum += cos * (rate)
            lower_sum += abs(cos)
    if lower_sum == 0:
        return 0
    return upper_sum / lower_sum

# get top n songs for user
def get_top_songs(user):
    top_songs = []
    for song in songs:
        song_id = get_id(songs, song)
        if train_set[user, song_id] == 0:
            top_songs.append((get_rating(user, song_id), song))
    return sorted(top_songs, reverse=True)[:n]

def print_top_songs_for_user(user):
    top_songs = get_top_songs(user)
    print("Top 10 songs for user with ratings: ")
    for (rating, song) in top_songs:
        print(song, str(rating[0][0]))



