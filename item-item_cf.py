#!/usr/bin/python

"""
Item-item collaborative Filtering
1. compute the average rating of each user
2. compute the deviation from the average rating
for each user item pair
3. compute the item-item correlation
4. Remove the item-item pairs with correlation below a given certain threshold
5. Evaluate predictions on the test set and plot the RMSE error for
different similarity cutoff thresholds

"""

