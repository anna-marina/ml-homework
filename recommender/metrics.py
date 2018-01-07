import math
from model import get_rating


def rmse(user_song_matrix, test_data):
    rates_am = 0  # R_test
    sqr_sum = 0

    for user in range(user_song_matrix.shape[0]):
        rated_by_user = user_song_matrix[user].tocsr().indices
        for song in rated_by_user:
            if test_data[user, song] == 0:
                predicted = get_rating(user, song)
                real = user_song_matrix[user, song]
                rates_am += 1
                # sqr_sum = sum (f(u, i) - r_ui)^2)
                # real = f(u,i)
                # predictes=r_ui
                sqr_sum += math.pow((real - predicted), 2)

    return math.sqrt(sqr_sum / rates_am)


def mae(user_song_matrix, test_data):
    rates_am = 0
    sqr_sum = 0

    for user in range(user_song_matrix.shape[0]):
        rated_by_user = user_song_matrix[user].tocsr().indices
        for song in rated_by_user:
            if test_data[user, song] == 0:
                predicted = get_rating(user, song)
                real = user_song_matrix[user, song]
                rates_am += 1
                sqr_sum += math.fabs((real - predicted))

    return sqr_sum / rates_am

