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
    # For Z-Image-Turbo (Qwen3 encoder, up to 512 tokens): concrete camera details and small real-world
    # imperfections read as a photograph rather than a render.
    "photo": (
        # (No camera brand here: Z-Image drew the named camera into the frame.)
        "Professional food photograph taken with a macro lens at f/4. "
        "Warm, soft natural window light from the left with a gentle fill, realistic shadows. The food is the hero "
        "and fills most of the frame, seen from a slightly elevated three-quarter angle. Every texture is real and "
        "tactile: crisp edges, glistening fat and sauce, tiny crumbs and imperfections, natural irregular shapes. "
        "Shallow depth of field with a softly blurred restaurant background in warm wood tones. True-to-life "
        "colours, appetizing, freshly served, award-winning food magazine photography. No text, no watermark."
    ),
    # The best burgers on the board are tight crops on near-black with warm rim light.
    "studio": (
        "Professional advertising food photograph taken with a macro lens, a tight close-up where the food fills "
        # v4 lost on "a glowing ember-like artefact on the bun's edge" from asking for a glowing rim light.
        "almost the whole frame. Dark, moody charcoal background, soft warm key light from the side and a gentle "
        "natural backlight, realistic soft shadows, no glow or lens flare. Ultra-realistic tactile "
        "textures: crunchy caramelised crust, glistening juices, glossy melting sauce, tiny crumbs. Shallow depth of "
        "field, rich warm true-to-life colours, mouth-watering, award-winning commercial food photography. No "
        "text, no watermark."
    ),
    "izakaya": (
        "Professional food photograph taken with a macro lens at f/2.8, a slightly elevated three-quarter "
        "close-up where the dish fills most of the frame. Set on a dark wooden counter in a cosy ramen shop at "
        "night, with warm amber lamps glowing as soft round bokeh in the background. Warm side light makes the "
        "broth and toppings glisten. Realistic tactile textures, natural imperfections, rich warm true-to-life "
        "colours, mouth-watering, award-winning food magazine photography. No text, no watermark."
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
    # v4 lessons from the judge on v3 ("steam looks painted", "cheese-pickle zone muddled", "fries look off"):
    # no steam, a clear layer order, crisp fries.
    "burger2": (
        "A juicy double bacon cheeseburger on a dark slate board, seen at eye level. Clear layers from the bottom: "
        "a toasted brioche bun, crisp green lettuce, a thick slice of ripe tomato, a seared beef patty with a dark "
        "crust and slices of cheddar melting over its edges in glossy drips, a second patty with more melted "
        "cheddar, crispy bacon rashers, and a shiny golden sesame-seed top bun. Meat juices glisten. A pile of "
        "crispy golden shoestring fries with flaky salt beside it."
    ),
    # burger2 came out clean but tame (one patty, little melt); the judge's first criterion is hunger, and
    # "molten cheese dripping" is what won v3 its games. Keep the drips, keep it free of steam.
    "burger3": (
        "A towering bacon cheeseburger on a dark slate board, seen at eye level. Two thick seared beef patties "
        "with dark caramelised crusts are stacked, each draped in molten cheddar that drips down the sides in "
        "long glossy golden drips onto the bottom bun. Crispy bacon rashers, crisp green lettuce, a slice of "
        "ripe tomato, pickles, and a shiny toasted sesame-seed brioche bun. Meat juices glisten. Crispy golden "
        "fries with flaky salt beside it."
    ),
    # Ramen is what most 93%+ entries show; Z-Image's first ramens had thin broth in a plain white bowl.
    "ramen2": (
        "A steaming bowl of rich tonkotsu ramen in a dark speckled stoneware bowl that fills the frame. The broth "
        "is creamy, opaque and golden with glistening droplets of fat and swirls of red chili oil. Wooden "
        "chopsticks lift a thick bundle of glossy wavy noodles out of the broth. Four slices of torched chashu "
        "pork belly with caramelised edges are fanned on one side, a halved soy-marinated egg shows a jammy "
        "orange yolk, a sheet of nori stands at the back, with sliced green scallions, sesame seeds and bean "
        "sprouts. Real wisps of steam rise from the hot broth."
    ),
    # v6 (ramen2, 90.7%) lost points for "the hand at the right edge is awkwardly formed" and "the orange swirl
    # looks painted": keep everything else, show no hand, and only a few drops of chili oil.
    "ramen3": (
        "A steaming bowl of rich tonkotsu ramen in a dark speckled stoneware bowl that fills the frame. The broth "
        "is creamy, opaque and golden with glistening droplets of fat and a few small drops of red chili oil. A "
        "pair of wooden chopsticks enters from the top right corner, held out of frame with no hand visible, and "
        "lifts a thick bundle of glossy wavy noodles out of the broth. Torched chashu pork belly slices with "
        "caramelised edges are fanned on one side, a halved soy-marinated egg shows a jammy orange yolk, a sheet "
        "of nori stands at the back, with sliced green scallions, sesame seeds and bean sprouts. Real wisps of "
        "steam rise from the hot broth."
    ),
    # v7 (ramen3) lost its one game for "lifted noodles look slightly uniform and plasticky".
    "ramen4": (
        "A steaming bowl of rich tonkotsu ramen in a dark speckled stoneware bowl that fills the frame. The broth "
        "is creamy, opaque and golden with glistening droplets of fat and a few small drops of red chili oil. A "
        "pair of wooden chopsticks enters from the top right corner, held out of frame with no hand visible, and "
        "lifts springy, wavy, golden-yellow alkaline noodles high out of the broth, each strand distinct and "
        "glistening, broth dripping from them. Torched chashu pork belly slices with caramelised edges and soft "
        "translucent fat are fanned on one side, two halves of a soy-marinated egg show jammy orange yolks, a "
        "sheet of nori stands at the back, with sliced green scallions and sesame seeds. Real wisps of steam rise "
        "from the hot broth."
    ),
    "lava2": (
        "A warm chocolate lava cake on a matte dark plate, a spoon has just broken it open so thick glossy molten "
        "chocolate pours out and pools on the plate. A scoop of vanilla bean ice cream with visible vanilla "
        "specks is just starting to melt beside it, three fresh raspberries, a light dusting of cocoa powder and "
        "a few flakes of sea salt."
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
