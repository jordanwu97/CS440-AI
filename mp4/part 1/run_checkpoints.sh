python3 snake_main.py --snake_head_x=200 --snake_head_y=200 --food_x=80 --food_y=80 --Ne=40 --C=40 --gamma=0.7 --train_episodes=100 --test_episodes=1 --show_episodes=0

cd checkpoints

python3 checkpoint.py --checkpoint_num=1

cd ..

python3 snake_main.py --snake_head_x=200 --snake_head_y=200 --food_x=80 --food_y=80 --Ne=20 --C=60 --gamma=0.5 --train_episodes=100 --test_episodes=1 --show_episodes=0

cd checkpoints

python3 checkpoint.py --checkpoint_num=2

cd ..

python3 snake_main.py --snake_head_x=80 --snake_head_y=80 --food_x=200 --food_y=200 --Ne=40 --C=40 --gamma=0.7 --train_episodes=100 --test_episodes=1 --show_episodes=0

cd checkpoints

python3 checkpoint.py --checkpoint_num=3

cd ..