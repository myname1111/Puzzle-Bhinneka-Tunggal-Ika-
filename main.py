"""# Game architecture

This code is organized into entities, components, processes (systems), stages, resources, events, worlds, assets
Here is a brief explanation of each:
- Entities: Something which contains no other data than its ID which points to components associated with said entity. They are all simple classes
- Components: Components store data and are associated to an entity
- Processes: Processes, or as they're more commonly known, "systems", are functions which operate on all entities having some components. They are responsible for almost all of the logic in the code
- Stages: Stages are simply groups of processes, there are two main stages, event and render:
    - The event stage is simply the stage where processes that require events, be that native pygame events or the events that I will explain later, are run
    - The render stage is responsible for rendering all the sprites and text and whatever else that needs to be rendered. It also deals with clearing the screen
- Resources: Resources represent the current game state which is independent of any individual entity. By that, I mean that no entity is associated with said game state. Examples of resources include: The servers, the current level and level information, as well as special entities
- Events: Events, not to be confused with native pygame events, are things which are temporary and visible to all entities. They represent signals, flags, and can be broadcast. Some processes run only if an event was broadcast. Events are handled by the EventServer
- Worlds: Worlds are simply different stages or screen of the game. There are three main stages: The start menu, the main game, and the lose screen
- Assets: Assets are anything that needs to be loaded before hand and are too perfomance-intensive to move around. Thus they are handled by the AssetServer which gives an AssetId which points to the asset when an Asset is added to the AssetServer. The AssetId can be used as a lightweight pointer to the actual Asset
"""

import json
import os
import random
from copy import deepcopy
from dataclasses import dataclass
from math import floor
from typing import Any

import esper
import pygame
from pygame.font import Font
from pygame.surface import Surface

FPS = 60
RESOLUTION = 1280, 720
FULL_Y_OFFSET = 150
X_OFFSET = 100
Y_SIZE = 420
X_GAP = 100
MULT_EXP = 1.2
MULT_CONST = 15


"""A sprite is just an image"""


class Sprite:
    def __init__(self, image: str | pygame.Surface, depth: int = 0):
        """Intializes the sprite

        Keyword arguments:
        - Image: Can be either a string or a Surface. In the former case, it'd assume that it was a path to the image and load it
        - Depth: The depth or z value of the image (default: 0)
        """
        if type(image) is str:
            self.image: Surface = pygame.image.load(image)
        elif type(image) is Surface:
            self.image = image
        self.depth: int = depth
        self.w: int = self.image.get_width()
        self.h: int = self.image.get_height()


"""A pointer to an asset of type T"""


@dataclass
class AssetId[T]:
    id: int


"""A server to handle assets of type T"""


@dataclass
class AssetServer[T]:
    assets: list[T]

    def add(self, new: T) -> AssetId[T]:
        """Add an asset of type T. Returns the AssetId of the asset

        Keyword arguments:
        - new: The asset to be added
        """
        self.assets.append(new)
        return AssetId(len(self.assets) - 1)

    def get(self, idx: AssetId[T]) -> T:
        """Get the asset from the AssetId

        Keyword arguments:
        - idx: The AssetId
        """
        return self.assets[idx.id]


"""A special type of AssetServer to handle sprites"""


@dataclass
class SpriteServer:
    sprites: AssetServer[Sprite]

    def add(self, new: Sprite | str | Surface) -> AssetId[Sprite]:
        if type(new) is str or type(new) is Surface:
            new_sprite = Sprite(new)
        elif type(new) is Sprite:
            new_sprite = new
        else:
            assert False

        return self.sprites.add(new_sprite)

    def get(self, idx: AssetId[Sprite]) -> Sprite:
        return self.sprites.get(idx)


"""Represents an image on a spritesheet"""


@dataclass
class SpriteSheetId:
    sprite_sheet_id: AssetId[Sprite]
    rect: pygame.Rect


"""Represents anything that can be rendered"""


@dataclass
class Renderable:
    sprite_id: AssetId[Sprite] | SpriteSheetId


"""Represents the position of something on the screen"""


@dataclass
class ScreenPos:
    x: float
    y: float

    @staticmethod
    def from_tuple(tup: tuple[float, float]):
        """Creates a ScreenPos from a tuple

        Keyword arguments:
        - from_tuple: The tuple to create it from
        """
        return ScreenPos(*tup)


"""Represents the relevant information to generate a Level"""


@dataclass
class LevelInfo:
    image: str
    desc: str

    @staticmethod
    def from_dict(in_dict: dict[str, str]):
        """Creates the LevelInfo from a dictionary

        Keyword arguments:
        - in_dict: The dictionary to create it from
        """
        image = in_dict["image_name"]
        desc = in_dict["desc"]
        return LevelInfo(image, desc)


"""A level generator"""


@dataclass
class LevelsResource:
    levels: list[LevelInfo]
    batches: int = 0

    @staticmethod
    def from_json(path: str, shuffle=True, seed=0):
        """Load all the levels from a json file

        Keyword arguments:
        - path: Path to the json file
        - shuffle: Whether or not to shuffle the list (default: True)
        - seed: Sets the seed if it were to shuffle
        """
        with open(path) as file:
            in_str = file.read()
        in_list = json.loads(in_str)
        level_info_list = [
            LevelInfo.from_dict(level_info_dict) for level_info_dict in in_list
        ]
        if shuffle:
            random.seed(seed)
            random.shuffle(level_info_list)
        return LevelsResource(level_info_list)

    def __getitem__(self, idx: int) -> LevelInfo:
        total_len = len(self.levels) * (1 + self.batches)
        if idx >= total_len:
            self.batches += 1
            random.shuffle(self.levels)
        actual_idx = idx % len(self.levels)
        return self.levels[actual_idx]


"""The ingame level that is current being played"""


@dataclass
class LevelGridResource:
    tiles: list[tuple[int | None, int]]
    solved: int
    empty: tuple[int, int]
    incorrect: int
    level_size: int
    tile_width: float
    tile_height: float
    x_offset: float
    y_offset: float
    timer_duration: float

    @staticmethod
    def get_moves_from_level(level: int, level_size: int):
        """Gets the minimum amount of moves that the shuffler should make for each level

        Keyword arguments:
        - level: The current level
        - level_size: The current level's size
        """
        level_area = level_size**2
        moves = (level + 2) * level_area
        return min(moves, level_area * 16)

    @staticmethod
    def get_surroundings(level_size: int, pos: tuple[int, int]):
        """Gets the surrounding valid tiles

        Keyword arguments:
        - level_size: The size of the level
        - pos: The position to be checked
        """
        out = []
        if pos[0] > 0:
            out.append((pos[0] - 1, pos[1]))
        if pos[0] < level_size - 1:
            out.append((pos[0] + 1, pos[1]))
        if pos[1] > 0:
            out.append((pos[0], pos[1] - 1))
        if pos[1] < level_size - 1:
            out.append((pos[0], pos[1] + 1))

        return out

    @staticmethod
    def get_idx(row_size: int, pos: tuple[int, int]) -> int:
        """Gets the idx of a pos

        Keyword arguments:
        - row_size: The size of each row
        - pos: The pos
        """
        return pos[1] * row_size + pos[0]

    @staticmethod
    def shuffle(
        grid: list[int | None], moves: int, level_size: int
    ) -> tuple[list[int | None], tuple[int, int]]:
        """Shuffles the level

        Keyword arguments:
        - grid: The grid to be shuffled
        - moves: The amount of moves to make
        - level_size: THe size of the level
        """
        empty_slot = (0, 0)
        grid_old = deepcopy(grid)
        while grid_old == grid:
            for _ in range(moves):
                surroundings = LevelGridResource.get_surroundings(
                    level_size, empty_slot
                )
                new_pos = random.choice(surroundings)
                old_idx = LevelGridResource.get_idx(level_size, empty_slot)
                new_idx = LevelGridResource.get_idx(level_size, new_pos)
                grid[old_idx] = grid[new_idx]
                grid[new_idx] = None
                empty_slot = new_pos
        return (grid, empty_slot)

    @staticmethod
    def create_level(level: int, sprite_server: SpriteServer, levels: LevelsResource):
        """Creates a level

        Keyword arguments:
        - level: The current level
        - sprite_server: The sprite server
        - levels: The information for the levels
        """
        level_size = level // 4 + 2
        level_area = level_size**2
        level_image = levels[level].image

        solved_image = pygame.image.load(f"assets/{level_image}")
        y_scaling_factor = Y_SIZE / solved_image.get_height()
        solved_image = pygame.transform.scale_by(solved_image, y_scaling_factor / 2)
        solved_image_sprite_id = sprite_server.add(solved_image)

        solved = esper.create_entity()
        esper.add_component(solved, Renderable(solved_image_sprite_id))
        esper.add_component(solved, Centered())

        image = pygame.image.load(f"assets/{level_image}")

        x_offset = RESOLUTION[0] / 4 + solved_image.get_width() / 2 + X_OFFSET
        x_size = RESOLUTION[0] - x_offset - X_GAP
        x_scaling_factor = x_size / image.get_width()
        scaling_factor = min(x_scaling_factor, y_scaling_factor)
        image = pygame.transform.scale_by(image, scaling_factor)
        image_sprite = Sprite(image)
        image_sprite_id = sprite_server.add(image_sprite)
        tile_width, tile_height = (
            image_sprite.w / level_size,
            image_sprite.h / level_size,
        )

        y_offset = RESOLUTION[1] / 2 - image_sprite.h / 2
        esper.add_component(
            solved,
            ScreenPos(
                RESOLUTION[0] / 4,
                RESOLUTION[1] / 2 + image_sprite.h / 2 - solved_image.get_height() / 2,
            ),
        )

        idxs: list[None | int] = [None]
        for i in range(level_area - 1):
            idxs.append(i)

        moves = LevelGridResource.get_moves_from_level(level, level_size)
        random.seed(level)
        (idxs, empty_slot) = LevelGridResource.shuffle(idxs, moves, level_size)

        sprites = []
        incorrect = 0
        for i in range(level_area):
            idx = idxs[i]
            if idx is None:
                if i != 0:
                    incorrect += 1
                sprites.append((None, 0))
                continue
            idx += 1
            if i != idx:
                incorrect += 1
            x, y = i % level_size, i // level_size
            rect_x, rect_y = idx % level_size, idx // level_size
            start_x, start_y = rect_x * tile_width, rect_y * tile_height
            pos_x, pos_y = x * tile_width, y * tile_height
            rect = pygame.Rect(start_x, start_y, tile_width, tile_height)
            tile_sprite = SpriteSheetId(image_sprite_id, rect)
            tile_entity = esper.create_entity()
            esper.add_component(tile_entity, Renderable(tile_sprite))
            esper.add_component(
                tile_entity, ScreenPos(pos_x + x_offset, pos_y + y_offset)
            )
            sprites.append((tile_entity, idx))

        timer_duration_multiplier = MULT_EXP ** (-level)
        timer_duration = level_area * timer_duration_multiplier * MULT_CONST

        return LevelGridResource(
            sprites,
            solved,
            empty_slot,
            incorrect,
            level_size,
            tile_width,
            tile_height,
            x_offset,
            y_offset,
            timer_duration,
        )

    def __getitem__(self, pos: tuple[int, int]) -> tuple[int | None, int]:
        idx = LevelGridResource.get_idx(self.level_size, pos)
        return self.tiles[idx]

    def __setitem__(self, pos: tuple[int, int], value: tuple[int | None, int]):
        idx = LevelGridResource.get_idx(self.level_size, pos)
        self.tiles[idx] = value

    def is_correct(self, check_pos: tuple[int, int], for_pos: tuple[int, int]) -> bool:
        """Checks if a position is correct if it were in another position

        Keyword arguments:
        - check_pos: The position it will check to
        - for_pos: The position that is checked
        """
        correct_idx = LevelGridResource.get_idx(self.level_size, check_pos)
        actual_idx = self[for_pos][1]
        return correct_idx == actual_idx

    def from_screen_pos(
        self, screen_pos: ScreenPos | tuple[float, float]
    ) -> tuple[int, int]:
        """Converts from a ScreenPos to a tile position

        Keyword arguments:
        - screen_pos: The screen pos to convert
        """
        if type(screen_pos) is ScreenPos:
            screen_pos = (screen_pos.x, screen_pos.y)
        elif type(screen_pos) is tuple:
            screen_pos = screen_pos
        else:
            assert False
        level_pos = (screen_pos[0] - self.x_offset, screen_pos[1] - self.y_offset)
        tile_pos = (level_pos[0] // self.tile_width, level_pos[1] // self.tile_height)
        return (int(tile_pos[0]), int(tile_pos[1]))

    def to_screen_pos(self, tile_pos: tuple[int, int]) -> ScreenPos:
        """Converts from a tile position to a ScreenPos

        Keyword arguments:
        - tile_pos: The tile pos to convert
        """
        level_pos = (tile_pos[0] * self.tile_width, tile_pos[1] * self.tile_height)
        screen_pos = (level_pos[0] + self.x_offset, level_pos[1] + self.y_offset)
        return ScreenPos(*screen_pos)

    def swap(self, old_pos: tuple[int, int]):
        """Swaps a tile with the empty tile

        Keyword arguments:
        - old_pos: The position of the tile to be swapped
        """
        new_pos = self.empty
        offset = 0
        if self.is_correct(old_pos, old_pos):
            offset += 1
        if self.is_correct(old_pos, new_pos):
            offset -= 1
        if self.is_correct(new_pos, new_pos):
            offset += 1
        if self.is_correct(new_pos, old_pos):
            offset -= 1

        self.incorrect += offset

        entity = self[old_pos][0]
        if entity is None:
            assert False
        new_screen_pos = self.to_screen_pos(new_pos)
        esper.component_for_entity(entity, ScreenPos).x = new_screen_pos.x
        esper.component_for_entity(entity, ScreenPos).y = new_screen_pos.y
        self[new_pos] = self[old_pos]
        self[old_pos] = (None, 0)
        self.empty = old_pos

    def delete(self):
        """Deletes all the entities associated with the level"""
        for ent, _ in self.tiles:
            if ent is None:
                continue
            esper.add_component(ent, Deleted())
        esper.add_component(self.solved, Deleted())

    def update(
        self, new_level: int, sprite_server: SpriteServer, levels: LevelsResource
    ):
        """Updates it to a new level

        Keyword arguments:
        - new_level: The new level
        - sprite_server: The sprite server
        - levels: The resource containing the information for the levels
        """
        self.delete()
        return LevelGridResource.create_level(new_level, sprite_server, levels)


"""A component representing text"""


@dataclass
class Text:
    text: str
    font_id: AssetId[Font]
    color: tuple[int, int, int] = (255, 255, 255)
    antialias: bool = True

    def render(self, font_server: AssetServer[Font]) -> Surface:
        """Render the text to a Surface

        Keyword arguments:
        - font_server: The font server
        """
        font = font_server.get(self.font_id)
        return font.render(self.text, self.antialias, self.color)


"""A component. Any entity with this component will be centered"""


class Centered:
    pass


"""This event is broadcast when the level is done, i.e, all the tiles are in place"""


@dataclass
class LevelDoneEvent:
    pass


"""An event"""


@dataclass
class Event[T]:
    data: T


"""This class manages events"""


@dataclass
class Events:
    buf1: dict[type[Any], list[Event[Any]]]
    buf2: dict[type[Any], list[Event[Any]]]
    broadcast_to_1: bool

    def broadcast(self, event: object):
        """Broadcast an event

        Keyword arguments:
        - event: The event to be broadcast
        """
        events = self.buf1 if self.broadcast_to_1 else self.buf2

        if type(event) in events:
            events[type(event)].append(Event(event))
        else:
            events[type(event)] = [Event(event)]

    def clear(self):
        """Updates the event. Clears all old events"""
        if self.broadcast_to_1:
            self.buf2 = {}
        else:
            self.buf1 = {}
        self.broadcast_to_1 = not self.broadcast_to_1

    def get[T](self, event_type: type[T]) -> list[Event[T]]:
        """Gets all events of a certain type"""
        events = self.buf2 if self.broadcast_to_1 else self.buf1
        try:
            return events[event_type]
        except KeyError:
            return []


"""Represents something that is clickable. Broadcasts an event when clicked"""


@dataclass
class Clickable[T]:
    on_click_event: T


"""An event which is broadcast when we continue to the nect level"""


@dataclass
class NextLevelEvent:
    pass


"""An entity with this component is part of the end UI, and will subsequently be deleted after it transitions to the next level"""


@dataclass
class EndUi:
    pass


"""An entity to be deleted"""


@dataclass
class Deleted:
    pass


"""Operations to change the world"""


class Operations:
    QUIT = 0
    CONTINUE = 1
    PROGRESS = 2

    @dataclass
    class Lose:
        level: int


"""An event that will change the world"""


@dataclass
class OperationEvent:
    operation: int


"""The Highest levels reached"""


@dataclass
class HighestLevels:
    absolute_highest_level: int

    @staticmethod
    def from_json(path: str):
        """Loads from a json file"""
        with open(path) as file:
            in_str = file.read()

        in_dict = json.loads(in_str)
        absolute_highest_level = in_dict["absolute_highest_level"]
        return HighestLevels(absolute_highest_level)

    def to_json(self, path: str):
        """Save to a json file"""
        out_str = json.dumps({"absolute_highest_level": self.absolute_highest_level})
        with open(path, "w") as file:
            file.write(out_str)

    def update(self, new: int):
        """Update the highest levels reached"""
        self.absolute_highest_level = max(self.absolute_highest_level, new)


"""The playlist in charged of playing the music"""


@dataclass
class Playlist:
    fn: list[str]
    nfn: list[str]
    playlist: list[str]
    idx: int
    is_fn: bool
    batches: int

    @staticmethod
    def load(path: str, seed: int):
        """Loads the play list from a path

        Keyword arguments:
        - path: The path to the playlist
        - seed: The seed to which it is shuffled
        """
        random.seed(seed)
        fn = os.listdir(f"{path}/FN/")
        fn = [f"assets/music/FN/{song}" for song in fn]
        nfn = os.listdir(f"{path}/NFN/")
        nfn = [f"assets/music/NFN/{song}" for song in nfn]

        random.shuffle(fn)
        random.shuffle(nfn)
        pygame.mixer.music.load(fn[0])
        pygame.mixer.music.queue(fn[1])
        return Playlist(fn, nfn, fn, 0, True, 0)

    def update(self):
        """Updates the song"""
        self.idx += 1
        if self.is_fn and self.idx >= len(self.playlist) - 1:
            self.playlist.extend(self.nfn)
        if self.is_fn:
            pygame.mixer.music.queue(self.playlist[self.idx + 1])
            return
        total_len = len(self.playlist) * (1 + self.batches)
        actual_idx = self.idx - total_len
        if actual_idx >= len(self.playlist) - 1:
            random.shuffle(self.playlist)
        pygame.mixer.music.queue(self.playlist[self.idx + 1])


def center_pos(uncentered_pos: ScreenPos, size: tuple[float, float]) -> ScreenPos:
    """Centers a position"""
    return ScreenPos(uncentered_pos.x - size[0] / 2, uncentered_pos.y - size[1] / 2)


def get_component_optional[T](entity: int, component: type[T]) -> T | None:
    """Get the component for an entity. If it doesn't exist returns None"""
    try:
        out_component = esper.component_for_entity(entity, component)
        return out_component
    except KeyError:
        return None


def get_size(
    renderable: Renderable, sprite_server: SpriteServer
) -> tuple[float, float]:
    """Gets the size of a renderable thing"""
    if type(renderable.sprite_id) is AssetId:
        sprite = sprite_server.get(renderable.sprite_id)
        return sprite.w, sprite.h
    elif type(renderable.sprite_id) is SpriteSheetId:
        return renderable.sprite_id.rect.size
    else:
        assert False


def render_sprites_process(fake_screen: Surface, sprite_server: SpriteServer):
    """Renders all renderables"""
    for ent, (renderable, screen_pos) in esper.get_components(Renderable, ScreenPos):
        centered = get_component_optional(ent, Centered)
        size = get_size(renderable, sprite_server)

        if type(renderable.sprite_id) is AssetId:
            sprite = sprite_server.get(renderable.sprite_id)
            pos = screen_pos if centered is None else center_pos(screen_pos, size)
            pos = (pos.x, pos.y)
            fake_screen.blit(sprite.image, pos)

        elif type(renderable.sprite_id) is SpriteSheetId:
            spritesheet = sprite_server.get(renderable.sprite_id.sprite_sheet_id)
            pos = screen_pos if centered is None else center_pos(screen_pos, size)
            pos = (pos.x, pos.y)
            fake_screen.blit(spritesheet.image, pos, renderable.sprite_id.rect)


def render_text_process(fake_screen: Surface, font_server: AssetServer[Font]):
    """Renders all the text"""
    for ent, (text, screen_pos) in esper.get_components(Text, ScreenPos):
        centered = get_component_optional(ent, Centered)
        surface = text.render(font_server)
        size = (surface.get_width(), surface.get_height())
        pos = screen_pos if centered is None else center_pos(screen_pos, size)
        pos = (pos.x, pos.y)
        fake_screen.blit(surface, pos)


def render_stage(
    window: Surface,
    fake_screen: Surface,
    sprite_server: SpriteServer,
    font_server: AssetServer[Font],
    background_color: tuple[int, int, int],
):
    """The render stage"""
    window.fill((0, 0, 0))
    fake_screen.fill(background_color)
    render_sprites_process(fake_screen, sprite_server)
    render_text_process(fake_screen, font_server)
    scaled = pygame.transform.smoothscale(fake_screen, window.get_size())
    window.blit(scaled, (0, 0))
    pygame.display.flip()


def move_event(
    mouse_pos: tuple[float, float],
    level: LevelGridResource,
    events: Events,
):
    """Moves a tile when clicked on"""
    tile_pos = level.from_screen_pos(mouse_pos)
    if (
        tile_pos[0] < 0
        or tile_pos[0] >= level.level_size
        or tile_pos[1] < 0
        or tile_pos[1] >= level.level_size
    ):
        return

    positions = LevelGridResource.get_surroundings(level.level_size, tile_pos)
    if level.empty not in positions:
        return
    old_pos = tile_pos
    level.swap(old_pos)
    if level.incorrect == 0:
        events.broadcast(LevelDoneEvent())


def level_done_process(
    events: Events,
    silkscreen_med: AssetId[Font],
    next_button_image: AssetId[Sprite],
    levels: LevelsResource,
    level_num: int,
):
    """Created the end ui"""
    level_done_events = events.get(LevelDoneEvent)
    if len(level_done_events) == 0:
        return

    level_text = esper.create_entity()
    esper.add_component(level_text, Text(levels[level_num].desc, silkscreen_med))
    esper.add_component(
        level_text, ScreenPos(RESOLUTION[0] / 2, Y_SIZE + FULL_Y_OFFSET + 40)
    )
    esper.add_component(level_text, Centered())
    esper.add_component(level_text, EndUi())

    button = esper.create_entity()
    esper.add_component(button, Renderable(next_button_image))
    esper.add_component(
        button, ScreenPos(RESOLUTION[0] / 2, Y_SIZE + FULL_Y_OFFSET + 100)
    )
    esper.add_component(button, Centered())
    esper.add_component(button, Clickable(NextLevelEvent()))
    esper.add_component(button, EndUi())


def handle_clicks_process(
    events: Events, mouse_pos: tuple[float, float], sprite_server: SpriteServer
):
    """Handle clicks. All clickable entities will broadcast their events"""
    for ent, (renderable, screen_pos, clickable) in esper.get_components(
        Renderable, ScreenPos, Clickable
    ):
        centered = get_component_optional(ent, Centered)
        width, height = size = get_size(renderable, sprite_server)
        centered_pos = screen_pos if centered is None else center_pos(screen_pos, size)
        aabb = pygame.Rect(centered_pos.x, centered_pos.y, width, height)
        if aabb.collidepoint(mouse_pos[0], mouse_pos[1]):
            events.broadcast(clickable.on_click_event)


def next_level_process(
    events: Events,
    level_num: int,
    level_text: int,
    level: LevelGridResource,
    sprite_server: SpriteServer,
    levels: LevelsResource,
    time_start: float,
) -> tuple[int, LevelGridResource, float]:
    """Move to the next level"""
    next_level_events = events.get(NextLevelEvent)
    if len(next_level_events) == 0:
        return level_num, level, time_start
    next_level = level_num + 1
    esper.component_for_entity(level_text, Text).text = f"Level: {next_level + 1}"
    for ent, _ in esper.get_component(EndUi):
        esper.add_component(ent, Deleted())
    level = level.update(next_level, sprite_server, levels)
    time_start = pygame.time.get_ticks() / 1000
    return next_level, level, time_start


def event_stage(
    event: pygame.event.Event,
    running: bool,
    window: Surface,
    level_grid_resource: LevelGridResource,
    events: Events,
    silkscreen_med: AssetId[Font],
    next_button_image: AssetId[Sprite],
    sprite_server: SpriteServer,
    level_num: int,
    levels: LevelsResource,
    level_text: int,
    time_start: float,
    playlist: Playlist,
) -> tuple[bool, int, LevelGridResource, float]:
    """Handles all events for the main game"""
    if event.type == pygame.QUIT:
        running = False
    if event.type == pygame.MOUSEBUTTONDOWN:
        scale = (
            RESOLUTION[0] / window.get_width(),
            RESOLUTION[1] / window.get_height(),
        )
        mouse_pos = pygame.mouse.get_pos()
        mouse_pos = (mouse_pos[0] * scale[0], mouse_pos[1] * scale[1])
        move_event(mouse_pos, level_grid_resource, events)
        handle_clicks_process(events, mouse_pos, sprite_server)
    if event.type == pygame.USEREVENT:
        playlist.update()
    level_done_process(events, silkscreen_med, next_button_image, levels, level_num)
    level_num, level_grid_resource, time_start = next_level_process(
        events,
        level_num,
        level_text,
        level_grid_resource,
        sprite_server,
        levels,
        time_start,
    )
    events.clear()
    return running, level_num, level_grid_resource, time_start


def deletion_process():
    """Delete all entities marked as deleted"""
    deletion_stack = [ent for ent, _ in esper.get_component(Deleted)]
    for ent in deletion_stack:
        esper.delete_entity(ent, immediate=True)


def get_time_left(level: LevelGridResource, time_start: float):
    """Get the time left"""
    time_elapsed = pygame.time.get_ticks() / 1000 - time_start
    time_left = level.timer_duration - time_elapsed
    return time_left


def update_timer(level: LevelGridResource, time_start: float, time_text: int):
    """Update the timer's text"""
    time_left = get_time_left(level, time_start)
    display_time = floor(time_left * 100) / 100.0
    esper.component_for_entity(time_text, Text).text = f"Time Left: {display_time}"


def lose_process(
    level: LevelGridResource, time_start: float, level_num: int
) -> int | Operations.Lose:
    """Change to the lose screen once there is no time left"""
    time_left = get_time_left(level, time_start)
    if time_left <= 0:
        return Operations.Lose(level_num)
    else:
        return Operations.CONTINUE


def main_game(
    window: pygame.Surface, fake_screen: Surface, playlist: Playlist
) -> int | Operations.Lose:
    """The main game"""
    clock = pygame.time.Clock()
    background_color = (51, 88, 114)

    sprite_server = SpriteServer(AssetServer([]))
    font_server = AssetServer([])
    silkscreen_large = font_server.add(Font("assets/slkscr.ttf", 50))
    silkscreen_med = font_server.add(Font("assets/slkscr.ttf", 35))
    events = Events({}, {}, True)

    next_button_image = sprite_server.add("assets/next_button.png")

    levels = LevelsResource.from_json("assets/images.json")
    level = 0
    level_grid_resource = LevelGridResource.create_level(level, sprite_server, levels)
    level_text = esper.create_entity()
    esper.add_component(level_text, Text("Level: 1", silkscreen_large))
    esper.add_component(level_text, ScreenPos(RESOLUTION[0] / 2, FULL_Y_OFFSET / 2))
    esper.add_component(level_text, Centered())

    time_start = pygame.time.get_ticks() / 1000
    time_text = esper.create_entity()
    esper.add_component(time_text, Text(f"Time Left: {time_start}", silkscreen_med))
    esper.add_component(time_text, ScreenPos(0, 0))

    running = True
    while running:
        for event in pygame.event.get():
            running, level, level_grid_resource, time_start = event_stage(
                event,
                running,
                window,
                level_grid_resource,
                events,
                silkscreen_med,
                next_button_image,
                sprite_server,
                level,
                levels,
                level_text,
                time_start,
                playlist,
            )

        update_timer(level_grid_resource, time_start, time_text)
        operation = lose_process(level_grid_resource, time_start, level)
        if operation != Operations.CONTINUE:
            return operation
        render_stage(window, fake_screen, sprite_server, font_server, background_color)
        deletion_process()
        clock.tick()

    return Operations.QUIT


def lose_event_stage(
    event: pygame.event.Event,
    running: bool,
    window: Surface,
    events: Events,
    sprite_server: SpriteServer,
    playlist: Playlist,
) -> bool:
    """The event stage for the lose world"""
    if event.type == pygame.QUIT:
        running = False
    if event.type == pygame.MOUSEBUTTONDOWN:
        scale = (
            RESOLUTION[0] / window.get_width(),
            RESOLUTION[1] / window.get_height(),
        )
        mouse_pos = pygame.mouse.get_pos()
        mouse_pos = (mouse_pos[0] * scale[0], mouse_pos[1] * scale[1])
        handle_clicks_process(events, mouse_pos, sprite_server)
    if event.type == pygame.USEREVENT:
        playlist.update()

    return running


def operation_process(events: Events) -> int:
    """Does an operation given an event"""
    operations = events.get(OperationEvent)
    if len(operations) > 0:
        return operations[-1].data.operation
    return Operations.CONTINUE


def lose(
    window: pygame.Surface,
    fake_screen: Surface,
    last_level: int,
    highest_levels: HighestLevels,
    playlist: Playlist,
) -> int:
    """The lose world"""
    clock = pygame.time.Clock()
    background_color = (216, 33, 33)

    sprite_server = SpriteServer(AssetServer([]))
    font_server = AssetServer([])
    silkscreen_large = font_server.add(Font("assets/slkscr.ttf", 100))
    silkscreen_med = font_server.add(Font("assets/slkscr.ttf", 50))
    events = Events({}, {}, True)

    lose_text = esper.create_entity()
    esper.add_component(lose_text, Text("YOU LOSE!", silkscreen_large))
    esper.add_component(lose_text, ScreenPos(RESOLUTION[0] / 2, RESOLUTION[1] / 4))
    esper.add_component(lose_text, Centered())

    level_text = esper.create_entity()
    esper.add_component(level_text, Text(f"Last Level: {last_level}", silkscreen_med))
    esper.add_component(
        level_text, ScreenPos(RESOLUTION[0] / 2, RESOLUTION[1] / 2 - 50)
    )
    esper.add_component(level_text, Centered())

    hs_text = esper.create_entity()
    esper.add_component(
        hs_text,
        Text(
            f"Absolute Highest Level: {highest_levels.absolute_highest_level}",
            silkscreen_med,
        ),
    )
    esper.add_component(hs_text, ScreenPos(RESOLUTION[0] / 2, RESOLUTION[1] / 2))
    esper.add_component(hs_text, Centered())

    quit_button = esper.create_entity()
    esper.add_component(
        quit_button, Renderable(sprite_server.add("assets/quit_button.png"))
    )
    esper.add_component(
        quit_button, ScreenPos(RESOLUTION[0] / 2 - 200, RESOLUTION[1] / 2 + 150)
    )
    esper.add_component(quit_button, Centered())
    esper.add_component(quit_button, Clickable(OperationEvent(Operations.QUIT)))

    retry_button = esper.create_entity()
    esper.add_component(
        retry_button, Renderable(sprite_server.add("assets/retry_button.png"))
    )
    esper.add_component(
        retry_button, ScreenPos(RESOLUTION[0] / 2 + 200, RESOLUTION[1] / 2 + 150)
    )
    esper.add_component(retry_button, Centered())
    esper.add_component(retry_button, Clickable(OperationEvent(Operations.PROGRESS)))

    running = True
    while running:
        for event in pygame.event.get():
            running = lose_event_stage(
                event, running, window, events, sprite_server, playlist
            )

        operation = operation_process(events)
        if operation != Operations.CONTINUE:
            return operation
        render_stage(window, fake_screen, sprite_server, font_server, background_color)
        deletion_process()
        clock.tick()
        events.clear()

    return Operations.QUIT


def start_event_stage(
    event: pygame.event.Event,
    events: Events,
    window: Surface,
    sprite_server: SpriteServer,
    playlist: Playlist,
    running: bool,
) -> bool:
    """The event loop for the start world"""
    if event.type == pygame.QUIT:
        running = False
    if event.type == pygame.MOUSEBUTTONDOWN:
        scale = (
            RESOLUTION[0] / window.get_width(),
            RESOLUTION[1] / window.get_height(),
        )
        mouse_pos = pygame.mouse.get_pos()
        mouse_pos = (mouse_pos[0] * scale[0], mouse_pos[1] * scale[1])
        handle_clicks_process(events, mouse_pos, sprite_server)
    if event.type == pygame.USEREVENT:
        playlist.update()
    return running


def start(
    window: Surface, fake_screen: Surface, playlist: Playlist
) -> int | Operations.Lose:
    """The start world"""
    clock = pygame.time.Clock()
    background_color = (51, 88, 114)

    sprite_server = SpriteServer(AssetServer([]))
    font_server = AssetServer([])
    silkscreen_large = font_server.add(Font("assets/slkscr.ttf", 50))
    events = Events({}, {}, True)

    lose_text = esper.create_entity()
    esper.add_component(
        lose_text, Text("Puzzle Bhineka Tunggal Inka", silkscreen_large)
    )
    esper.add_component(lose_text, ScreenPos(RESOLUTION[0] / 2, RESOLUTION[1] / 4))
    esper.add_component(lose_text, Centered())

    quit_button = esper.create_entity()
    esper.add_component(
        quit_button, Renderable(sprite_server.add("assets/start_button.png"))
    )
    esper.add_component(quit_button, ScreenPos(RESOLUTION[0] / 2, RESOLUTION[1] / 2))
    esper.add_component(quit_button, Centered())
    esper.add_component(quit_button, Clickable(OperationEvent(Operations.PROGRESS)))

    running = True
    while running:
        for event in pygame.event.get():
            running = start_event_stage(
                event, events, window, sprite_server, playlist, running
            )

        operation = operation_process(events)
        if operation != Operations.CONTINUE:
            return operation
        render_stage(window, fake_screen, sprite_server, font_server, background_color)
        deletion_process()
        clock.tick()
        events.clear()

    return Operations.QUIT


def main():
    pygame.init()
    window = pygame.display.set_mode(RESOLUTION, pygame.RESIZABLE)
    fake_screen = Surface(RESOLUTION)
    highest_levels = HighestLevels.from_json("data/score_data.json")
    pygame.mixer.music.set_endevent(pygame.USEREVENT)
    playlist = Playlist.load("assets/music", 3141)
    pygame.mixer.music.play()

    while True:
        esper.switch_world("start")
        operation = start(window, fake_screen, playlist)
        esper.switch_world("default")
        esper.delete_world("start")
        if operation == Operations.QUIT:
            break
        operation = main_game(window, fake_screen, playlist)
        if operation == Operations.QUIT:
            break
        if type(operation) is Operations.Lose:
            highest_levels.update(operation.level)
            esper.switch_world("lose")
            esper.delete_world("default")
            operation = lose(
                window, fake_screen, operation.level, highest_levels, playlist
            )
        if operation == Operations.QUIT:
            break
        esper.switch_world("default")
        esper.delete_world("lose")
    highest_levels.to_json("data/score_data.json")


if __name__ == "__main__":
    main()
