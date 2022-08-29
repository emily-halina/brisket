import os


def main():
    desired_episode = 45
    os.chdir(os.getcwd() + "\\models\\counterAgent100")
    dir_list = sorted(os.listdir(), key=lambda t: os.stat(t).st_mtime)

    b = 0
    e = 0
    for i in range(desired_episode):
        b += 15
        e += 15

    b -= 3

    print(dir_list[b:e])

main()
