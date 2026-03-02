const SAMPLE_MENU = [
  "Grilled Chicken Wrap",
  "Roasted Vegetables",
  "Pasta Primavera",
  "Garden Salad",
  "Black Bean Burger",
  "Tomato Basil Soup",
  "Brown Rice Pilaf",
  "Fresh Fruit Bar",
  "Veggie Stir Fry",
  "Lemon Herb Fish",
  "Margherita Flatbread",
  "Caesar Salad",
];

interface MenuPanelProps {
  title?: string;
  subtitle?: string;
}

export function MenuPanel({
  title = "Today's Menu",
  subtitle = "Snapshot of featured items",
}: MenuPanelProps) {
  const midpoint = Math.ceil(SAMPLE_MENU.length / 2);
  const leftColumn = SAMPLE_MENU.slice(0, midpoint);
  const rightColumn = SAMPLE_MENU.slice(midpoint);

  return (
    <section className="flex h-full flex-col rounded-3xl border border-border bg-card/80 p-5 shadow-sm sm:p-6 lg:p-7">
      <header className="mb-4 flex flex-col gap-1 sm:mb-6">
        <h2 className="text-lg font-semibold tracking-tight sm:text-xl">
          {title}
        </h2>
        <p className="text-sm text-muted-foreground">{subtitle}</p>
      </header>

      <div className="relative -mx-1 flex-1 overflow-hidden">
        <div className="h-full overflow-auto px-1">
          <div className="grid grid-cols-1 gap-x-10 gap-y-2 text-sm sm:grid-cols-2 sm:gap-y-3">
            <ul className="space-y-1">
              {leftColumn.map((item) => (
                <li
                  key={item}
                  className="flex items-center justify-between rounded-lg px-2 py-1.5 text-foreground/90 hover:bg-muted/80"
                >
                  <span>{item}</span>
                </li>
              ))}
            </ul>
            <ul className="mt-2 space-y-1 sm:mt-0">
              {rightColumn.map((item) => (
                <li
                  key={item}
                  className="flex items-center justify-between rounded-lg px-2 py-1.5 text-foreground/90 hover:bg-muted/80"
                >
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </section>
  );
}

