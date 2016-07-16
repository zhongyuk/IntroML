#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    ### your code goes here
    from math import ceil
    removal = int(ceil(len(ages)*.1))
    errors = [abs(n-p) for p, n in zip(predictions, net_worths)]
    zip_data = zip(ages, net_worths, errors)
    zip_data.sort(key=lambda x: x[2], reverse=True)
    cleaned_data = zip_data[removal:]
    
    return cleaned_data

