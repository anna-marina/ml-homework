import math
from model import get_rating


def rmse(train_data, test_data):
    rates_am = 0  # R_test
    sqr_sum = 0

    for user in range(train_data.shape[0]):
        rated_by_user = train_data[user].tocsr().indices
        for song in rated_by_user:
            if test_data[user, song] == 0:
                predicted = get_rating(user, song)
                real = train_data[user, song]
                rates_am += 1
                # sqr_sum = sum (f(u, i) - r_ui)^2)
                # real = f(u,i)
                # predictes=r_ui
                sqr_sum += math.pow((real - predicted), 2)

    return math.sqrt(sqr_sum / rates_am)


def mae(train_data, test_data):
    rates_am = 0
    sqr_sum = 0

    for user in range(train_data.shape[0]):
        rated_by_user = train_data[user].tocsr().indices
        for song in rated_by_user:
            if test_data[user, song] == 0:
                predicted = get_rating(user, song)
                real = train_data[user, song]
                rates_am += 1
                sqr_sum += math.fabs((real - predicted))

    return sqr_sum / rates_am


def dcg(train_data, test_data, idealFlag):
    # N - top recommendation count
    # j - song position
    j = 0
    rates_am = 0
    sum = 0.0
    for user in range(train_data.shape[0]):
        rated_by_user = train_data[user].tocsr().indices
        for song in rated_by_user:
            real = train_data[user, song]
            j += 1
            rates_am += 1
            rate_ = get_rating(user, song)
            d = max(float(math.log(j, 2)), 1.0)
            if idealFlag:
                sum += real / d
            else:
                sum += rate_ / d
        j = 0

    if rates_am == 0:
        return 0
    else:
        return sum / rates_am


def ndcg(train_data, test_data):
    dcg_ = dcg(train_data, test_data, False)
    ideal_dcg = dcg(train_data, test_data, True)
    if ideal_dcg == 0:
        return 0
    else:
        result = dcg_ / ideal_dcg
        return result[0, 0]
