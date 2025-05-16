def exp_moving_average(data, alpha=0.9):
    if not data:
        return []
    output = [data[0]]
    for value in data[1:]:
        avg = alpha*value + (1-alpha)*output[-1]
        output.append(avg)
    return output

data = [i for i in range(1, 100)]
output = exp_moving_average(data)
print(output)