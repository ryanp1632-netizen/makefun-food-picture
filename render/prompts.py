"""Food-photography prompts shared by the generators.

FLUX.1-schnell reads at most 256 T5 tokens and CLIP only the first 77, so each prompt leads with the dish
and keeps the photographic style short.
"""

STYLES = {
    "dark": (
        "Close-up editorial food photography, low three-quarter angle, the dish fills the frame, shallow depth "
        "of field, dark moody background with warm bokeh, golden backlight, glossy highlights, mouth-watering."
    ),
    # The judge's strongest house images were believable, cleanly lit food in soft daylight, and it marked our
    # dark, high-contrast burger down as "plasticky" and "AI sheen". This style asks for a real photograph.
    "window": (
        "Real candid food photograph taken with a full-frame camera and 85mm lens, soft natural window light from "
        "the side, gentle shadows, true-to-life colours, crisp natural texture, shallow depth of field, the dish "
        "fills the frame, fresh and mouth-watering, nothing artificial."
    ),
}
STYLE = STYLES["dark"]

DISHES = {
    "ramen": (
        "Steaming bowl of tonkotsu ramen, a wide speckled ceramic bowl filling the frame, full to the brim. "
        "A pair of wooden chopsticks from the upper right lifts glossy golden noodles high above the rich amber "
        "broth. Neatly fanned slices of seared chashu pork, a halved ramen egg with a jammy orange yolk, a "
        "standing sheet of nori, green scallions, sesame seeds, clouds of white steam rising."
    ),
    "burger": (
        "Towering double cheeseburger on a dark slate board: glossy toasted brioche bun with sesame seeds, two "
        "seared smash patties with crispy edges, molten cheddar dripping down the side, crispy bacon, lettuce, "
        "tomato, pickles, golden fries beside it, a wisp of steam."
    ),
    "pizza": (
        "Neapolitan margherita pizza on a wooden peel, a slice lifted away stretching long strings of melted "
        "mozzarella. Puffy leopard-spotted charred crust, bright tomato sauce, fresh basil, olive oil drizzle, "
        "wood-fired oven glowing in the background."
    ),
    "lava": (
        "Chocolate lava cake on a black plate, cut open with molten glossy chocolate flowing out, a scoop of "
        "vanilla bean ice cream melting beside it, fresh raspberries, cocoa dust, sea salt flakes, a curl of steam."
    ),
    "steak": (
        "Thick seared ribeye steak sliced to show a juicy pink medium-rare centre on a dark wooden board, herb "
        "garlic butter melting over the slices, rosemary, roasted garlic, flaky salt, steam rising."
    ),
    "pancakes": (
        "Tall stack of fluffy golden pancakes, maple syrup pouring and cascading down the sides, a melting pat of "
        "butter on top, fresh blueberries and raspberries, powdered sugar."
    ),
    "carbonara": (
        "Bowl of spaghetti carbonara twirled into a glossy nest, silky creamy egg and pecorino sauce coating every "
        "strand, crispy golden guanciale pieces, freshly cracked black pepper, a shower of grated cheese, a fork "
        "resting on the side, steam rising."
    ),
    "tart": (
        "Slice of fresh strawberry tart on a white plate, buttery golden crumbly pastry, thick vanilla custard "
        "cream, glossy glazed ripe strawberries piled high, a few mint leaves, dusting of icing sugar."
    ),
    "salmon": (
        "Pan-seared salmon fillet with crispy golden skin and a flaky juicy pink centre on a white plate, lemon "
        "butter sauce glistening, a lemon wedge, fresh dill, tender asparagus, steam rising."
    ),
    "cinnamon": (
        "Warm gooey cinnamon roll fresh from the oven, thick cream cheese icing melting and dripping down the "
        "swirled golden layers, cinnamon sugar glistening, on a small plate, steam rising."
    ),
}
