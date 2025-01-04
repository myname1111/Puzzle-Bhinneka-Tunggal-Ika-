import random
from dataclasses import dataclass
from typing import Any

import esper
import pygame
from pygame.font import Font
from pygame.surface import Surface

FPS = 60
RESOLUTION = 1280, 720
Y_OFFSET = 150
Y_SIZE = 420


class Sprite:
    def __init__(self, image: str | pygame.Surface, depth: int = 0):
        if type(image) is str:
            self.image: Surface = pygame.image.load(image)
        elif type(image) is Surface:
            self.image = image
        self.depth: int = depth
        self.w: int = self.image.get_width()
        self.h: int = self.image.get_height()


@dataclass
class AssetId[T]:
    id: int


@dataclass
class AssetServer[T]:
    assets: list[T]

    def add(self, new: T) -> AssetId[T]:
        self.assets.append(new)
        return AssetId(len(self.assets) - 1)

    def get(self, idx: AssetId[T]) -> T:
        return self.assets[idx.id]


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


@dataclass
class SpriteSheetId:
    sprite_sheet_id: AssetId[Sprite]
    rect: pygame.Rect


@dataclass
class Renderable:
    sprite_id: AssetId[Sprite] | SpriteSheetId


@dataclass
class ScreenPos:
    x: float
    y: float

    @staticmethod
    def from_tuple(tup: tuple[float, float]):
        return ScreenPos(*tup)


@dataclass
class LevelGridResource:
    tiles: list[tuple[int | None, int]]
    empty: tuple[int, int]
    incorrect: int
    level_size: int
    tile_width: float
    tile_height: float
    level: int

    @staticmethod
    def get_moves_from_level(level: int, level_size: int):
        level_area = level_size**2
        moves = (level + 2) * level_area
        return min(moves, level_area * 16)

    @staticmethod
    def get_surroundings(level_size: int, pos: tuple[int, int]):
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
        return pos[1] * row_size + pos[0]

    @staticmethod
    def shuffle(
        grid: list[int | None], moves: int, level_size: int
    ) -> tuple[list[int | None], tuple[int, int]]:
        empty_slot = (0, 0)
        for _ in range(moves):
            surroundings = LevelGridResource.get_surroundings(level_size, empty_slot)
            new_pos = random.choice(surroundings)
            old_idx = LevelGridResource.get_idx(level_size, empty_slot)
            new_idx = LevelGridResource.get_idx(level_size, new_pos)
            grid[old_idx] = grid[new_idx]
            grid[new_idx] = None
            empty_slot = new_pos
        return (grid, empty_slot)

    @staticmethod
    def create_level(level: int, sprite_server: SpriteServer):
        level_size = level // 4 + 2
        level_area = level_size**2

        image = pygame.image.load(f"assets/{level + 1}.png")
        scaling_factor = Y_SIZE / image.get_height()
        image = pygame.transform.scale_by(image, scaling_factor)
        image_sprite = Sprite(image)
        image_sprite_id = sprite_server.add(image_sprite)
        tile_width, tile_height = (
            image_sprite.w / level_size,
            image_sprite.h / level_size,
        )
        idxs: list[None | int] = [None]
        empty = (
            RESOLUTION[0] - image_sprite.w
        )  # Yes, I know this can go negative. No, I don't care
        for i in range(level_area - 1):
            idxs.append(i)

        moves = LevelGridResource.get_moves_from_level(level, level_size)
        random.seed(level)
        (idxs, empty_slot) = LevelGridResource.shuffle(idxs, moves, level_size)
        print(idxs)

        sprites = []
        incorrect = 0
        x_offset = empty / 2
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
                tile_entity, ScreenPos(pos_x + x_offset, pos_y + Y_OFFSET)
            )
            sprites.append((tile_entity, idx))

        return LevelGridResource(
            sprites, empty_slot, incorrect, level_size, tile_width, tile_height, level
        )

    def __getitem__(self, pos: tuple[int, int]) -> tuple[int | None, int]:
        idx = LevelGridResource.get_idx(self.level_size, pos)
        return self.tiles[idx]

    def __setitem__(self, pos: tuple[int, int], value: tuple[int | None, int]):
        idx = LevelGridResource.get_idx(self.level_size, pos)
        self.tiles[idx] = value

    def get_x_offset(self) -> float:
        full = self.tile_width * self.level_size
        empty = RESOLUTION[0] - full
        return empty / 2

    def is_correct(self, check_pos: tuple[int, int], for_pos: tuple[int, int]) -> bool:
        correct_idx = LevelGridResource.get_idx(self.level_size, check_pos)
        actual_idx = self[for_pos][1]
        return correct_idx == actual_idx

    def from_screen_pos(
        self, screen_pos: ScreenPos | tuple[float, float]
    ) -> tuple[int, int]:
        if type(screen_pos) is ScreenPos:
            screen_pos = (screen_pos.x, screen_pos.y)
        elif type(screen_pos) is tuple:
            screen_pos = screen_pos
        else:
            assert False
        x_offset = self.get_x_offset()
        level_pos = (screen_pos[0] - x_offset, screen_pos[1] - Y_OFFSET)
        tile_pos = (level_pos[0] // self.tile_width, level_pos[1] // self.tile_height)
        return (int(tile_pos[0]), int(tile_pos[1]))

    def to_screen_pos(self, tile_pos: tuple[int, int]) -> ScreenPos:
        level_pos = (tile_pos[0] * self.tile_width, tile_pos[1] * self.tile_height)
        x_offset = self.get_x_offset()
        screen_pos = (level_pos[0] + x_offset, level_pos[1] + Y_OFFSET)
        return ScreenPos(*screen_pos)

    def swap(self, old_pos: tuple[int, int]):
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


@dataclass
class Text:
    text: str
    font_id: AssetId[Font]
    color: tuple[int, int, int] = (255, 255, 255)
    antialias: bool = True

    def render(self, font_server: AssetServer[Font]) -> Surface:
        font = font_server.get(self.font_id)
        return font.render(self.text, self.antialias, self.color)


class Centered:
    pass


@dataclass
class LevelDoneEvent:
    current_level: int


@dataclass
class Event[T]:
    data: T


@dataclass
class Events:
    buf1: dict[type[Any], list[Event[Any]]]
    buf2: dict[type[Any], list[Event[Any]]]
    broadcast_to_1: bool

    def broadcast[T](self, event: T):
        events = self.buf1 if self.broadcast_to_1 else self.buf2

        if type(event) in events:
            events[type(event)].append(Event(event))
        else:
            events[type(event)] = [Event(event)]

    def clear(self):
        if self.broadcast_to_1:
            self.buf2 = {}
        else:
            self.buf1 = {}
        self.broadcast_to_1 = not self.broadcast_to_1

    def get[T](self, event_type: type[T]) -> list[Event[T]]:
        events = self.buf2 if self.broadcast_to_1 else self.buf1
        try:
            return events[event_type]
        except KeyError:
            return []


@dataclass
class Clickable[T]:
    on_click_event: T


@dataclass
class NextLevelEvent:
    next_level: int


def center_pos(uncentered_pos: ScreenPos, size: tuple[float, float]) -> ScreenPos:
    return ScreenPos(uncentered_pos.x - size[0] / 2, uncentered_pos.y - size[1] / 2)


def get_component_optional(entity: int, component):
    try:
        out_component = esper.component_for_entity(entity, component)
        return out_component
    except KeyError:
        return None


def get_size(
    renderable: Renderable, sprite_server: SpriteServer
) -> tuple[float, float]:
    if type(renderable.sprite_id) is AssetId:
        sprite = sprite_server.get(renderable.sprite_id)
        return sprite.w, sprite.h
    elif type(renderable.sprite_id) is SpriteSheetId:
        return renderable.sprite_id.rect.size
    else:
        assert False


def render_sprites_process(fake_screen: Surface, sprite_server: SpriteServer):
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
):
    window.fill((0, 0, 0))
    fake_screen.fill((0, 0, 0))
    render_sprites_process(fake_screen, sprite_server)
    render_text_process(fake_screen, font_server)
    scaled = pygame.transform.smoothscale(fake_screen, window.get_size())
    window.blit(scaled, (0, 0))
    pygame.display.flip()


def move_event(
    mouse_pos: tuple[float, float], level: LevelGridResource, events: Events
):
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
        events.broadcast(LevelDoneEvent(level.level))


def level_done_process(
    events: Events, silkscreen_med: AssetId[Font], next_button_image: AssetId[Sprite]
):
    level_done_events = events.get(LevelDoneEvent)
    if len(level_done_events) == 0:
        return
    current_level = level_done_events[0].data.current_level

    level_text = esper.create_entity()
    esper.add_component(level_text, Text("Rumah gadang", silkscreen_med))
    esper.add_component(
        level_text, ScreenPos(RESOLUTION[0] / 2, Y_SIZE + Y_OFFSET + 40)
    )
    esper.add_component(level_text, Centered())

    button = esper.create_entity()
    esper.add_component(button, Renderable(next_button_image))
    esper.add_component(button, ScreenPos(RESOLUTION[0] / 2, Y_SIZE + Y_OFFSET + 100))
    esper.add_component(button, Centered())
    esper.add_component(button, Clickable(NextLevelEvent(current_level + 1)))


def handle_clicks_process(
    events: Events, mouse_pos: tuple[float, float], sprite_server: SpriteServer
):
    for ent, (renderable, screen_pos, clickable) in esper.get_components(
        Renderable, ScreenPos, Clickable
    ):
        centered = get_component_optional(ent, Centered)
        width, height = size = get_size(renderable, sprite_server)
        centered_pos = screen_pos if centered is None else center_pos(screen_pos, size)
        aabb = pygame.Rect(centered_pos.x, centered_pos.y, width, height)
        if aabb.collidepoint(mouse_pos[0], mouse_pos[1]):
            events.broadcast(clickable.on_click_event)


def next_level_process(events: Events):
    next_level_events = events.get(NextLevelEvent)
    if len(next_level_events) == 0:
        return
    next_level = next_level_events[0].data.next_level
    print(f"next level: {next_level}")


def event_stage(
    event: pygame.event.Event,
    running: bool,
    window: Surface,
    level_grid_resource: LevelGridResource,
    events: Events,
    silkscreen_med: AssetId[Font],
    next_button_image: AssetId[Sprite],
    sprite_server: SpriteServer,
) -> bool:
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
    level_done_process(events, silkscreen_med, next_button_image)
    next_level_process(events)
    events.clear()
    return running


def main():
    pygame.init()
    window = pygame.display.set_mode(RESOLUTION, pygame.RESIZABLE)
    fake_screen = Surface(RESOLUTION)
    clock = pygame.time.Clock()

    sprite_server = SpriteServer(AssetServer([]))
    font_server = AssetServer([])
    silkscreen_large = font_server.add(Font("assets/slkscr.ttf", 50))
    silkscreen_med = font_server.add(Font("assets/slkscr.ttf", 35))
    events = Events({}, {}, True)

    next_button_image = sprite_server.add("assets/next_button.png")

    level = 0
    level_grid_resource = LevelGridResource.create_level(level, sprite_server)
    level_text = esper.create_entity()
    esper.add_component(level_text, Text("Level: 1", silkscreen_large))
    esper.add_component(level_text, ScreenPos(RESOLUTION[0] / 2, Y_OFFSET / 2))
    esper.add_component(level_text, Centered())

    running = True
    while running:
        for event in pygame.event.get():
            running = event_stage(
                event,
                running,
                window,
                level_grid_resource,
                events,
                silkscreen_med,
                next_button_image,
                sprite_server,
            )

        render_stage(window, fake_screen, sprite_server, font_server)
        clock.tick()


if __name__ == "__main__":
    main()
