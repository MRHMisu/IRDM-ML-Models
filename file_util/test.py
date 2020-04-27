validation_file_path = "../dataset/validation_data.tsv"
traninign_file_path = "../dataset/train_data.tsv"


def read_lines(path):
    file_reader = open(path, 'r')
    lines = file_reader.readlines()
    file_reader.close()
    return lines


def read_n_lines(path, n):
    count = 0;
    list = []
    file_reader = open(path, 'r')
    while count < n:
        line = file_reader.readline()
        list.append(line)
        count += 1
    file_reader.close()
    return list


def test():
    lines = read_lines(traninign_file_path)
    count = 0;
    counter = 0
    for line in lines:
        if counter != 0:
            elements = line.split("\t")
            relevancy = float(elements[4].rstrip())
            # if relevancy == 1.0:
            count += 1
        counter += 1
    print(count)


if __name__ == "__main__":
    test()

# training -> Total=4364339  ,relevant= 4797, non-relevant=4359542
# validation->Total=1103039  , relevant= 1208    non-relevant=1101831
