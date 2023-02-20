from snake_game import SnakeGame, Direction

def play_game(network):
    game = SnakeGame()
    score = 0

    while not game.game_over:
        # Observe the current state of the game
        head_x, head_y = game.head
        food_x, food_y = game.food
        body = game.body
        state = [head_x, head_y, food_x, food_y]
        for segment in body:
            state += [segment[0], segment[1]]

        # Pad the state with zeros if the snake is shorter than 4 segments
        while len(state) < 24:
            state += [0, 0]

        # Use the network to choose an action
        output = network.activate(state)
        action = output.index(max(output))

        # Convert the action to a direction
        if action == 0:
            direction = Direction.UP
        elif action == 1:
            direction = Direction.DOWN
        elif action == 2:
            direction = Direction.LEFT
        elif action == 3:
            direction = Direction.RIGHT

        # Take the action and get the new state and score
        game.play_step(direction)
        new_score = game.score
        if new_score > score:
            score = new_score

    return score
