"""Food-photography prompts shared by the generators.

FLUX.1-schnell reads at most 256 T5 tokens and CLIP only the first 77, so each prompt leads with the dish
and keeps the photographic style short.
"""

STYLE = (
    "Close-up editorial food photography, low three-quarter angle, the dish fills the frame, shallow depth of "
    "field, dark moody background with warm bokeh, golden backlight, glossy highlights, mouth-watering."
)

DISHES = {
    "ramen": (
        "Steaming bowl of tonkotsu ramen in a dark stoneware bowl on a rustic wooden table. Wooden chopsticks "
        "lift glossy golden noodles from creamy broth. Torched chashu pork with charred edges, a halved ramen egg "
        "with a jammy orange yolk, green scallions, sesame seeds, nori, red chili oil, white steam rising."
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
}
