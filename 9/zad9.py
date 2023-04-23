#!/usr/bin/env python3
# Based on https://python101.readthedocs.io/pl/latest/pygame/pong/#
import pygame
from typing import Type
import skfuzzy as fuzz
import skfuzzy.control as fuzzcontrol

FPS = 30


class Board:
    def __init__(self, width: int, height: int):
        self.surface = pygame.display.set_mode((width, height), 0, 32)
        pygame.display.set_caption("AIFundamentals - PongGame")

    def draw(self, *args):
        background = (0, 0, 0)
        self.surface.fill(background)
        for drawable in args:
            drawable.draw_on(self.surface)

        pygame.display.update()


class Drawable:
    def __init__(self, x: int, y: int, width: int, height: int, color=(255, 255, 255)):
        self.width = width
        self.height = height
        self.color = color
        self.surface = pygame.Surface(
            [width, height], pygame.SRCALPHA, 32
        ).convert_alpha()
        self.rect = self.surface.get_rect(x=x, y=y)

    def draw_on(self, surface):
        surface.blit(self.surface, self.rect)


class Ball(Drawable):
    def __init__(
        self, x: int, y: int, radius: int = 20, color=(255, 10, 0), speed: int = 3,
    ):
        super(Ball, self).__init__(x, y, radius, radius, color)
        pygame.draw.ellipse(self.surface, self.color, [0, 0, self.width, self.height])
        self.x_speed = speed
        self.y_speed = speed
        self.start_speed = speed
        self.start_x = x
        self.start_y = y
        self.start_color = color
        self.last_collision = 0

    def bounce_y(self):
        self.y_speed *= -1

    def bounce_x(self):
        self.x_speed *= -1

    def bounce_y_power(self):
        self.color = (
            self.color[0],
            self.color[1] + 10 if self.color[1] < 255 else self.color[1],
            self.color[2],
        )
        pygame.draw.ellipse(self.surface, self.color, [0, 0, self.width, self.height])
        self.x_speed *= 1.1
        self.y_speed *= 1.1
        self.bounce_y()

    def reset(self):
        self.rect.x = self.start_x
        self.rect.y = self.start_y
        self.x_speed = self.start_speed
        self.y_speed = self.start_speed
        self.color = self.start_color
        self.bounce_y()

    def move(self, board: Board, *args):
        self.rect.x += round(self.x_speed)
        self.rect.y += round(self.y_speed)

        if self.rect.x < 0 or self.rect.x > (
            board.surface.get_width() - self.rect.width
        ):
            self.bounce_x()

        if self.rect.y < 0 or self.rect.y > (
            board.surface.get_height() - self.rect.height
        ):
            self.reset()

        timestamp = pygame.time.get_ticks()
        if timestamp - self.last_collision < FPS * 4:
            return

        for racket in args:
            if self.rect.colliderect(racket.rect):
                self.last_collision = pygame.time.get_ticks()
                if (self.rect.right < racket.rect.left + racket.rect.width // 4) or (
                    self.rect.left > racket.rect.right - racket.rect.width // 4
                ):
                    self.bounce_y_power()
                else:
                    self.bounce_y()


class Racket(Drawable):
    def __init__(
        self,
        x: int,
        y: int,
        width: int = 80,
        height: int = 20,
        color=(255, 255, 255),
        max_speed: int = 10,
    ):
        super(Racket, self).__init__(x, y, width, height, color)
        self.max_speed = max_speed
        self.surface.fill(color)

    def move(self, x: int, board: Board):
        delta = x - self.rect.x
        delta = self.max_speed if delta > self.max_speed else delta
        delta = -self.max_speed if delta < -self.max_speed else delta
        delta = 0 if (self.rect.x + delta) < 0 else delta
        delta = (
            0
            if (self.rect.x + self.width + delta) > board.surface.get_width()
            else delta
        )
        
        # print("X: " +str(self.rect.x))
        # print(delta)


        self.rect.x += delta


class Player:
    def __init__(self, racket: Racket, ball: Ball, board: Board) -> None:
        self.ball = ball
        self.racket = racket
        self.board = board

    def move(self, x: int):
        self.ball.rect.centery
        self.racket.move(x, self.board)

    def move_manual(self, x: int):
        """
        Do nothing, control is defined in derived classes
        """
        pass

    def act(self, x_diff: int, y_diff: int):
        """
        Do nothing, control is defined in derived classes
        """
        pass


class PongGame:
    def __init__(
        self, width: int, height: int, player1: Type[Player], player2: Type[Player]
    ):
        pygame.init()
        self.board = Board(width, height)
        self.fps_clock = pygame.time.Clock()
        self.ball = Ball(width // 2, height // 2)

        self.opponent_paddle = Racket(x=width // 2, y=0)
        self.oponent = player1(self.opponent_paddle, self.ball, self.board)

        self.player_paddle = Racket(x=width // 2, y=height - 20)
        self.player = player2(self.player_paddle, self.ball, self.board)

    def run(self):
        while not self.handle_events():
            self.ball.move(self.board, self.player_paddle, self.opponent_paddle)
            self.board.draw(
                self.ball, self.player_paddle, self.opponent_paddle,
            )
            self.oponent.act(
                self.oponent.racket.rect.centerx - self.ball.rect.centerx,
                self.oponent.racket.rect.centery - self.ball.rect.centery,
            )
            self.player.act(
                self.player.racket.rect.centerx - self.ball.rect.centerx,
                self.player.racket.rect.centery - self.ball.rect.centery,
            )
            self.fps_clock.tick(FPS)

    def handle_events(self):
        for event in pygame.event.get():
            if (event.type == pygame.QUIT) or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                pygame.quit()
                return True
        keys = pygame.key.get_pressed()
        if keys[pygame.constants.K_LEFT]:
            self.player.move_manual(0)
        elif keys[pygame.constants.K_RIGHT]:
            self.player.move_manual(self.board.surface.get_width())
        return False


class NaiveOponent(Player):
    def __init__(self, racket: Racket, ball: Ball, board: Board):
        super(NaiveOponent, self).__init__(racket, ball, board)

    def act(self, x_diff: int, y_diff: int):
        x_cent = self.ball.rect.centerx
        self.move(x_cent)


class HumanPlayer(Player):
    def __init__(self, racket: Racket, ball: Ball, board: Board):
        super(HumanPlayer, self).__init__(racket, ball, board)

    def move_manual(self, x: int):
        # print(self.ball.rect.centery)
        self.move(x)


# ----------------------------------
# DO NOT MODIFY CODE ABOVE THIS LINE
# ----------------------------------


class FuzzyPlayer(Player):
    def __init__(self, racket: Racket, ball: Ball, board: Board):
        super(FuzzyPlayer, self).__init__(racket, ball, board)
        
        # self.fuzzy_controler(1,1)
        # self.fuzzy_controler()
        # x_dist = fuzz.control.Antecedent...
        # y_dist = fuzz.control.Antecedent...
        # speed = fuzz.control.Consequent...
        # racket_controller = fuzz.control.ControlSystem...

    def act(self, x_diff: int, y_diff: int):
        velocity = self.make_decision(x_diff, y_diff)
        self.move(self.racket.rect.x + velocity)

    def fuzzy_controler(self,x,y):
        import numpy as np
        # x_distance = fuzzcontrol.Antecedent((abs(self.racket.rect.centerx-self.ball.rect.centerx),1), 'x_distance')
        # y_distance = fuzzcontrol.Antecedent((abs(self.racket.rect.centery-self.ball.rect.centery),1), 'y_distance')
        # movement = fuzzcontrol.Consequent(np.arange(0, 1, 1), 'movement')
        # print("x:"+ str(self.racket.rect.centerx - self.ball.rect.centerx))
        # print("y:" + str(self.racket.rect.centery - self.ball.rect.centery))
        x = self.racket.rect.centerx - self.ball.rect.centerx
        y = self.racket.rect.centery - self.ball.rect.centery
        # if x==0:
        #     x+=1
        x_distance = fuzzcontrol.Antecedent(np.arange(-800, 801, 1), 'x_distance')
        y_distance = fuzzcontrol.Antecedent(np.arange(1, 401, 1), 'y_distance')
        movement = fuzzcontrol.Consequent(np.arange(-10,11, 1), 'movement')

        
        # x_distance = fuzzcontrol.Antecedent(np.arange(0, 801, 1), 'x_distance')
        # y_distance = fuzzcontrol.Antecedent(np.arange(0, 401, 1), 'y_distance')
        # movement = fuzzcontrol.Consequent(np.arange(-400, 401, 1), 'movement')

        # Auto-membership function population
        #prawdopodobnie ręcznie wypadało by to zrobić
        x_distance['poor'] = fuzz.trapmf(x_distance.universe, [-800,-800,-100  , 0])
        x_distance['average'] = fuzz.trapmf(x_distance.universe, [-200,-100, 100 , 200])
        x_distance['good'] = fuzz.trapmf(x_distance.universe, [0,100, 800 , 800])
        # x_distance.automf(3)
        x_distance.view()

        # x_distance.view()
        y_distance['poor'] = fuzz.trapmf(y_distance.universe, [0,100,200,200])
        # y_distance['average'] = fuzz.trapmf(y_distance.universe, [100,180,220,300])
        y_distance['average'] = fuzz.trimf(y_distance.universe, [190,200,210])
        y_distance['good'] = fuzz.trapmf(y_distance.universe, [200,200,300,400])
        # y_distance.view()
        # y_distance.automf(3)
        # y_distance.view()


        # Custom membership functions can be built interactively with a familiar,
        # Pythonic API
        # movement['right'] = fuzz.trimf(movement.universe, [-10,-5,0])
        # movement['stay'] = fuzz.trimf(movement.universe, [-1,0,1])
        # movement['left'] = fuzz.trimf(movement.universe, [0,5,10])

        # movement.view()
        movement['right'] = fuzz.trapmf(movement.universe, [-10,-10, -5 , 0])
        movement['stay'] = fuzz.trapmf(movement.universe, [-1, -1, 1, 1])
        movement['left'] = fuzz.trapmf(movement.universe, [0,5, 10, 10])
        
        #na razie max mamy 3.47 jakimś cudem

        # movement.view()
        # movement['left'] = fuzz.trapmf(movement.universe, [-400, -400, -200, -100])
        # movement['stay'] = fuzz.trapmf(movement.universe, [-100, -50, 50, 100])
        # movement['right'] = fuzz.trapmf(movement.universe, [100, 200, 400, 400])
        
        # # movement.view()
        # You can see how these look with .view()
        # x_distance['average'].view()
        # y_distance['average'].view()
        # movement.view()
        

        # Create the rules
        rule1_1 = fuzzcontrol.Rule(x_distance['poor'] & y_distance['poor'], movement['left'])
        rule1_2 = fuzzcontrol.Rule(x_distance['poor'] & y_distance['average'], movement['left'])
        rule1_3 = fuzzcontrol.Rule(x_distance['poor'] & y_distance['good'], movement['left'])
        rule2_1 = fuzzcontrol.Rule(x_distance['average'], movement['stay'])
        rule2_2 = fuzzcontrol.Rule(x_distance['average'], movement['stay'])
        rule2_3 = fuzzcontrol.Rule(x_distance['average'], movement['stay'])
        rule3_1 = fuzzcontrol.Rule(x_distance['good'] & y_distance['poor'], movement['right'])
        rule3_2 = fuzzcontrol.Rule(x_distance['good'] & y_distance['average'], movement['right'])
        rule3_3 = fuzzcontrol.Rule(x_distance['good'] & y_distance['good'], movement['right'])

        # rule1 = fuzzcontrol.Rule(x_distance['good'], movement['right'])
        # rule2 = fuzzcontrol.Rule(x_distance['average'], movement['stay'])
        # rule3 = fuzzcontrol.Rule(x_distance['poor'], movement['left'])
 
        # rule1.view()

        # Create the control system
        tipping_ctrl = fuzzcontrol.ControlSystem([rule1_1,rule1_2,rule1_3,rule2_1,rule2_2,rule2_3, rule3_1,rule3_2, rule3_3])
        # tipping_ctrl = fuzzcontrol.ControlSystem([rule1,rule2,rule3])

        # Create a tipping simulation
        tipping = fuzzcontrol.ControlSystemSimulation(tipping_ctrl)

        # Pass inputs to the ControlSystem using Antecedent labels with Pythonic API
        # Note: if you like passing many inputs all at once, use .inputs(dict_of_data)
        tipping.input['x_distance'] = x
        tipping.input['y_distance'] = y

        # tipping.input['y_distance'] = y
        # Crunch the numbers
        tipping.compute()
        # print(tipping.output)
        # movement.view(sim=tipping)

        # print(tipping.output['movement'])
        return tipping.output['movement']*3

    def make_decision(self, x_diff: int, y_diff: int):
        # racket_controller.compute()
        # ...
        
        res = self.fuzzy_controler(self.racket.rect.centerx,self.racket.rect.centery)
        # print(res)
        
        #czyli mamy przedział -10- 0 i 0 10
        return res


if __name__ == "__main__":
    # game = PongGame(800, 400, NaiveOponent, HumanPlayer)

    #sprawdzić rozmiary w input itp ponieważ  zawsze ujemene wychodzi a dystans powyżej 750 nie spadł nigdy

    game = PongGame(800, 400, NaiveOponent, FuzzyPlayer)
    game.run()