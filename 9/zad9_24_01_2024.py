#!/usr/bin/env python3
# Based on https://python101.readthedocs.io/pl/latest/pygame/pong/#
import pygame
from typing import Type
import skfuzzy as fuzz
import skfuzzy.control as fuzzcontrol
from skfuzzy import defuzz


FPS = 30
ilosc_klatek_wykonanych = 0

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
            print("right" + str(self.board.surface.get_width()))
            # print("diff" + self.board.surface.get_width())
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
        print(self.racket.rect.centerx)
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

        x_distance = fuzzcontrol.Antecedent(np.arange(-800, 801, 1), 'x_distance')
        y_distance = fuzzcontrol.Antecedent(np.arange(0, 401, 1), 'y_distance')
        # movement = fuzzcontrol.Consequent(np.arange(0,801, 1), 'movement')
        movement = fuzzcontrol.Consequent(np.arange(-800,801, 1), 'movement')
        # movement = fuzzcontrol.Consequent(np.arange(-10,11, 1), 'movement')
    
        x_distance['poor'] = fuzz.trapmf(x_distance.universe, [-800,-800,-700  , 0])
        # x_distance['average'] = fuzz.trapmf(x_distance.universe, [-200,-100, 100 , 200])
        x_distance['average'] = fuzz.trimf(x_distance.universe, [-700,0, 700])
        x_distance['good'] = fuzz.trapmf(x_distance.universe, [0,700, 800 , 800])
        # x_distance.view()

        
        y_distance['poor'] = fuzz.trapmf(y_distance.universe, [0,0,120,200])
        # y_distance['average'] = fuzz.trapmf(y_distance.universe, [100,180,220,300])
        y_distance['average'] = fuzz.trimf(y_distance.universe, [120,200,280])
        y_distance['good'] = fuzz.trapmf(y_distance.universe, [200,280,400,400])
        # y_distance.view()

        #te są jak dotąd the best
        movement['very_left'] = fuzz.trapmf(movement.universe, [-800,-800, -750 , -650])
        movement['left'] = fuzz.trapmf(movement.universe, [-750, -650, -600, -370])
        movement['slight_left'] = fuzz.trapmf(movement.universe, [-600, -370, -300, 0])
        movement['stay'] = fuzz.trimf(movement.universe, [-300, 0, 300])
        movement['slight_right'] = fuzz.trapmf(movement.universe, [0, 300, 370, 600])
        movement['right'] = fuzz.trapmf(movement.universe, [370, 600, 650, 750])
        movement['very_right'] = fuzz.trapmf(movement.universe, [650, 750, 800, 800])
        
        # movement.view()

        rule1 = fuzzcontrol.Rule(x_distance['good'] & y_distance['poor'], movement['very_left'])
        rule2 = fuzzcontrol.Rule(x_distance['good'] & y_distance['average'], movement['left'])
        rule3 = fuzzcontrol.Rule(x_distance['good'] & y_distance['good'], movement['slight_left'])
        rule4 = fuzzcontrol.Rule(x_distance['average'], movement['stay'])
        rule5 = fuzzcontrol.Rule(x_distance['poor'] & y_distance['good'], movement['slight_right'])
        rule6 = fuzzcontrol.Rule(x_distance['poor'] & y_distance['average'], movement['right'])
        rule7 = fuzzcontrol.Rule(x_distance['poor'] & y_distance['poor'], movement['very_right'])
 
        # Create the control system
        tipping_ctrl = fuzzcontrol.ControlSystem([rule1,rule2,rule3,rule4,rule5,rule6,rule7])
    
        # Create a tipping simulation
        tipping = fuzzcontrol.ControlSystemSimulation(tipping_ctrl)
    
        tipping.input['x_distance'] = x
        tipping.input['y_distance'] = y

        tipping.compute()
        # print(tipping.output)

        #wykresy
        x_distance.view()
        y_distance.view()
        movement.view()

        # print(tipping.output['movement'])
        # movement.view(sim=tipping)
        return tipping.output['movement']

    def make_decision(self, x_diff: int, y_diff: int):
        # racket_controller.compute()
        # ...
        # print(x_diff)
        # print(y_diff)
        #x_diff i y_diff to odległość środka paletki do środka piłki
        res = self.fuzzy_controler(x_diff,y_diff)

        return res


if __name__ == "__main__":
    # game = PongGame(800, 400, NaiveOponent, HumanPlayer)

    #sprawdzić rozmiary w input itp ponieważ  zawsze ujemene wychodzi a dystans powyżej 750 nie spadł nigdy

    game = PongGame(800, 400, NaiveOponent, FuzzyPlayer)
    game.run()