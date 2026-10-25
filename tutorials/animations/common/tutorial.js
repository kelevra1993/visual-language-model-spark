/*
 * Engine of the HTML scenes of the tutorial videos. The builder (tutorial_builder/animation_rendering.py) opens a page
 * of tutorials/animations in headless Chrome, calls Tutorial.setupScene(scene, timing) once with the scene of the
 * storyboard and the times of its sentences, then Tutorial.renderFrame(time) for every frame of the video, and
 * captures the page. The scene is an SVG diagram of elements (box, arrow, text, image, stack, bars, cuboid, grid,
 * curve) that appear, grow, are highlighted, dimmed and leave with the sentences of the narration, every change eased
 * over a fraction of a second. renderFrame returns whether something still moves, so that the builder captures a
 * still frame once. A page adds its own animated parts with Tutorial.registerCustom(setupFunction), the setup
 * returning the function that updates them at a time.
 */
(function () {
    "use strict";

    const SVG_NAMESPACE = "http://www.w3.org/2000/svg";
    const SCENE_WIDTH = 1760;
    const SCENE_HEIGHT = 740;

    // Colours of the frames and of the kinds of elements, the same as NAMED_COLOURS of tutorial_builder/layout.py
    const COLOURS = {
        text: [232, 232, 232], dim: [140, 150, 166], accent: [76, 201, 240], key: [125, 196, 255],
        value: [240, 200, 120], comment: [125, 135, 145], prompt: [120, 220, 120], card: [12, 19, 32],
        box: [21, 31, 50], border: [28, 40, 62], background: [6, 10, 18], terminal: [4, 7, 13],
        blue: [57, 135, 229], aqua: [25, 158, 112], yellow: [201, 133, 0], magenta: [213, 81, 129],
        violet: [144, 133, 233], red: [230, 103, 103], white: [255, 255, 255], black: [0, 0, 0]
    };

    // Durations of the transitions, in seconds of the video, and the opacity of a dimmed element
    const APPEAR_SECONDS = 0.45;
    const GROW_SECONDS = 0.6;
    const HIGHLIGHT_SECONDS = 0.25;
    const DIMMED_OPACITY = 0.3;

    // Speed of the dots flowing along a highlighted arrow, in pixels per second
    const FLOW_PIXELS_PER_SECOND = 600;

    // Line heights of the text styles of tutorial.css, used to place multiline texts
    const LINE_HEIGHTS = {
        diagram_heading: 42, diagram_bold: 36, diagram_text: 36, diagram_small: 29, diagram_tiny: 24,
        diagram_monospace: 32, diagram_monospace_small: 27, diagram_monospace_tiny: 22, code_text: 34
    };

    const engine = { scene: null, timing: null, root: null, records: [], customUpdates: [], customSetups: [] };

    /* Colour of an element: a name of COLOURS, a "#rrggbb" string or an RGB list. */
    function resolveColour(colour) {
        if (Array.isArray(colour) && colour.length === 3) {
            return colour.map(Number);
        }
        if (typeof colour === "string" && colour in COLOURS) {
            return COLOURS[colour];
        }
        if (typeof colour === "string" && /^#[0-9a-fA-F]{6}$/.test(colour)) {
            return [1, 3, 5].map((start) => parseInt(colour.slice(start, start + 2), 16));
        }
        throw new Error(`Unknown colour ${JSON.stringify(colour)}`);
    }

    /* CSS colour of an RGB list. */
    function cssColour(rgb) {
        return `rgb(${Math.round(rgb[0])}, ${Math.round(rgb[1])}, ${Math.round(rgb[2])})`;
    }

    /* Colour between two RGB lists, at a mix within [0, 1]. */
    function mixColour(first, second, mix) {
        return [0, 1, 2].map((channel) => first[channel] + (second[channel] - first[channel]) * mix);
    }

    /* Value clamped to [0, 1]. */
    function clampUnit(value) {
        return Math.min(1, Math.max(0, value));
    }

    /* Smooth step easing of a progress, clamped to [0, 1]. */
    function ease(progress) {
        const clamped = clampUnit(progress);
        return clamped * clamped * (3 - 2 * clamped);
    }

    /* Linear map of a value from a range onto another one. */
    function scaleValue(value, sourceStart, sourceEnd, targetStart, targetEnd) {
        const span = sourceEnd - sourceStart || 1;
        return targetStart + (value - sourceStart) / span * (targetEnd - targetStart);
    }

    /* Points of a polygon or polyline as an SVG points attribute. */
    function formatPoints(points) {
        return points.map((point) => `${point[0]},${point[1]}`).join(" ");
    }

    /* New SVG element with its attributes, appended to its parent. */
    function createSvgElement(tag, attributes, parent) {
        const element = document.createElementNS(SVG_NAMESPACE, tag);
        for (const [name, value] of Object.entries(attributes || {})) {
            element.setAttribute(name, String(value));
        }
        if (parent) {
            parent.appendChild(element);
        }
        return element;
    }

    /* Start of a sentence of the scene, in seconds from the scene start, 0 for the sentence -1 (the whole scene). */
    function sentenceStart(sentenceIndex) {
        if (sentenceIndex < 0) {
            return 0;
        }
        if (sentenceIndex >= engine.timing.sentences.length) {
            return Infinity;
        }
        return engine.timing.sentences[sentenceIndex].start;
    }

    /* Sentence being spoken, or last spoken, at a time of the scene, -1 before the first one. */
    function activeSentence(time) {
        let active = -1;
        engine.timing.sentences.forEach((sentence, index) => {
            if (time >= sentence.start) {
                active = index;
            }
        });
        return active;
    }

    /*
     * Value of a per-sentence state (highlighted, dimmed, shown) at a time: the state of the active sentence, eased
     * from the state of the previous sentence over the start of the active one. Returns the value and whether it moves.
     */
    function sentenceStateValue(sentenceIndices, time, offValue, onValue, durationSeconds) {
        const active = activeSentence(time);
        const indices = sentenceIndices || [];
        const currentValue = indices.includes(active) ? onValue : offValue;
        const previousValue = active >= 1 && indices.includes(active - 1) ? onValue : offValue;
        if (active < 0 || currentValue === previousValue) {
            return { value: currentValue, moving: false };
        }
        const progress = (time - sentenceStart(active)) / durationSeconds;
        return { value: previousValue + (currentValue - previousValue) * ease(progress), moving: progress < 1 };
    }

    /*
     * Appearance of an element at a time: its opacity (fading in when its appear sentence starts and out when the
     * sentence after its until sentence starts, at once for appear -1), its growth (arrows, bars, curves and grids
     * grown from their start over GROW_SECONDS, or over grow_seconds) and whether either still moves.
     */
    function computeAppearance(element, time) {
        const appearIndex = element.appear === undefined ? -1 : element.appear;
        const appearTime = sentenceStart(appearIndex);
        let opacity = appearIndex < 0 ? 1 : ease((time - appearTime) / APPEAR_SECONDS);
        let moving = appearIndex >= 0 && time >= appearTime && time < appearTime + APPEAR_SECONDS;
        if (element.until !== undefined && element.until !== null) {
            const leaveTime = sentenceStart(element.until + 1);
            opacity *= 1 - ease((time - leaveTime) / APPEAR_SECONDS);
            moving = moving || (time >= leaveTime && time < leaveTime + APPEAR_SECONDS);
        }
        const grows = element.grow !== undefined ? element.grow :
            ["arrow", "bars", "curve", "grid"].includes(element.type);
        const growSeconds = element.grow_seconds || GROW_SECONDS;
        let growth = 1;
        if (grows && appearIndex >= 0) {
            growth = ease((time - appearTime) / growSeconds);
            moving = moving || (time >= appearTime && time < appearTime + growSeconds);
        }
        if (time < appearTime) {
            opacity = 0;
        }
        return { opacity: opacity, growth: growth, moving: moving };
    }

    /* Fills an SVG text element with the lines of a text, placed for its anchor ("la", "mm", "ls", "rm"...). */
    function setMultilineText(textElement, text, x, y, anchor, fontClass) {
        const lines = String(text).split("\n");
        const lineHeight = LINE_HEIGHTS[fontClass] || 34;
        const horizontal = { l: "start", m: "middle", r: "end" }[anchor[0]] || "start";
        const baselines = { a: "text-before-edge", m: "central", s: "alphabetic", d: "text-after-edge" };
        const vertical = anchor[1] || "a";
        textElement.setAttribute("text-anchor", horizontal);
        textElement.setAttribute("dominant-baseline", baselines[vertical] || "text-before-edge");
        while (textElement.firstChild) {
            textElement.removeChild(textElement.firstChild);
        }

        // The block of lines placed so that its anchor point is the given point
        let firstY = y;
        if (vertical === "m") {
            firstY = y - (lines.length - 1) * lineHeight / 2;
        } else if (vertical === "s" || vertical === "d") {
            firstY = y - (lines.length - 1) * lineHeight;
        }
        lines.forEach((line, index) => {
            const lineElement = createSvgElement("tspan", { x: x, y: firstY + index * lineHeight }, textElement);
            lineElement.textContent = line;
        });
    }

    /* New text element of a font class and colour, filled with a text at a point and anchor. */
    function createText(parent, text, x, y, anchor, fontClass, colour) {
        const textElement = createSvgElement("text", { class: fontClass, fill: cssColour(resolveColour(colour)) },
                                             parent);
        setMultilineText(textElement, text, x, y, anchor, fontClass);
        return textElement;
    }

    /* Point at a distance along a polyline, with the direction of its segment. */
    function pointAlong(points, distance) {
        let remaining = distance;
        for (let index = 1; index < points.length; index++) {
            const [startX, startY] = points[index - 1];
            const [endX, endY] = points[index];
            const length = Math.hypot(endX - startX, endY - startY);
            if (remaining <= length || index === points.length - 1) {
                const fraction = length > 0 ? Math.min(1, remaining / length) : 0;
                return { x: startX + (endX - startX) * fraction, y: startY + (endY - startY) * fraction,
                         angle: Math.atan2(endY - startY, endX - startX) };
            }
            remaining -= length;
        }
        return { x: points[0][0], y: points[0][1], angle: 0 };
    }

    /* Total length of a polyline. */
    function polylineLength(points) {
        let length = 0;
        for (let index = 1; index < points.length; index++) {
            length += Math.hypot(points[index][0] - points[index - 1][0], points[index][1] - points[index - 1][1]);
        }
        return length;
    }

    /* Places the triangular head of an arrow with its tip at a point, pointing along an angle. */
    function placeArrowHead(polygon, tipX, tipY, angle, headLength) {
        const baseX = tipX - Math.cos(angle) * headLength;
        const baseY = tipY - Math.sin(angle) * headLength;
        const halfWidth = headLength * 0.5;
        polygon.setAttribute("points", formatPoints([[tipX, tipY],
            [baseX - Math.sin(angle) * halfWidth, baseY + Math.cos(angle) * halfWidth],
            [baseX + Math.sin(angle) * halfWidth, baseY - Math.cos(angle) * halfWidth]]));
    }

    /* Accent outline drawn around a box while its element is highlighted. */
    function createHighlightOutline(parent, left, top, right, bottom) {
        return createSvgElement("rect", { x: left - 8, y: top - 8, width: right - left + 16, height: bottom - top + 16,
                                          rx: 8, fill: "none", stroke: cssColour(COLOURS.accent), "stroke-width": 4,
                                          opacity: 0 }, parent);
    }

    /* Box: a rounded rectangle filled and outlined in its colours, its text centred; a null fill draws the outline
     * only, such as a frame over an image. */
    function buildBox(element, group) {
        const [left, top, right, bottom] = element.box;
        const fill = element.fill === null ? "none" : cssColour(resolveColour(element.fill || "box"));
        const rectangle = createSvgElement("rect", { x: left, y: top, width: right - left, height: bottom - top,
                                                     rx: element.radius === undefined ? 12 : element.radius,
                                                     fill: fill }, group);
        if (element.text) {
            createText(group, element.text, (left + right) / 2, (top + bottom) / 2, "mm",
                       element.font || "diagram_text", element.text_colour || "text");
        }
        return { rectangle: rectangle, outline: resolveColour(element.outline || "border") };
    }

    /* Box at a moment: its outline turning to the accent colour, and thicker, while highlighted. */
    function updateBox(parts, element, state) {
        const width = element.width || 3;
        parts.rectangle.setAttribute("stroke", cssColour(mixColour(parts.outline, COLOURS.accent, state.highlight)));
        parts.rectangle.setAttribute("stroke-width", width + state.highlight * (4 - width));
        return false;
    }

    /* Arrow: a line along its points with heads at its end, start, both or none, a label shown once grown, and dots
     * flowing along it while it is highlighted when flow is set. */
    function buildArrow(element, group) {
        const width = element.width || 4;
        const totalLength = polylineLength(element.points);

        // A plain arrow grows by the offset of a single dash as long as the line; a dashed one keeps its own dashes
        // and grows by a mask, a solid copy of the line whose single dash uncovers it
        const lineAttributes = { points: formatPoints(element.points), fill: "none", "stroke-width": width,
                                 "stroke-linecap": element.dash ? "butt" : "round", "stroke-linejoin": "round",
                                 "stroke-dasharray": element.dash ? element.dash.join(" ") :
                                     `${totalLength} ${totalLength}` };
        let growingLine = null;
        if (element.dash) {
            const maskIdentifier = `arrow-mask-${engine.records.length}-${Math.round(element.points[0][0])}-` +
                `${Math.round(element.points[0][1])}`;
            const mask = createSvgElement("mask", { id: maskIdentifier, maskUnits: "userSpaceOnUse", x: 0, y: 0,
                                                    width: SCENE_WIDTH, height: SCENE_HEIGHT }, group);
            growingLine = createSvgElement("polyline", { points: formatPoints(element.points), fill: "none",
                                                         stroke: "white", "stroke-width": width + 4,
                                                         "stroke-linejoin": "round",
                                                         "stroke-dasharray": `${totalLength} ${totalLength}` }, mask);
            lineAttributes.mask = `url(#${maskIdentifier})`;
        }
        const line = createSvgElement("polyline", lineAttributes, group);
        const head = element.head || "end";
        const parts = {
            line: line, growingLine: growingLine || line, totalLength: totalLength, headLength: 14 + 2 * width,
            endHead: ["end", "both"].includes(head) ? createSvgElement("polygon", {}, group) : null,
            startHead: ["start", "both"].includes(head) ? createSvgElement("polygon", {}, group) : null,
            label: null, flowDots: [], colour: resolveColour(element.colour || "dim")
        };
        if (element.label) {
            parts.label = createText(group, element.label, element.label_position[0], element.label_position[1],
                                     element.label_anchor || "mm", element.label_font || "diagram_small",
                                     element.label_colour || "dim");
        }
        if (element.flow) {
            for (let index = 0; index < 4; index++) {
                parts.flowDots.push(createSvgElement("circle", { r: width + 2, fill: cssColour(COLOURS.accent) },
                                                     group));
            }
        }
        return parts;
    }

    /* Arrow at a moment: grown up to its growth, its heads at the grown end, its colour turning to the accent while
     * highlighted, its flowing dots moving; it moves as long as dots flow. */
    function updateArrow(parts, element, state, time) {
        const colour = cssColour(mixColour(parts.colour, COLOURS.accent, state.highlight));
        const grownLength = parts.totalLength * state.appearance.growth;
        const headInset = parts.endHead ? parts.headLength * 0.7 : 0;
        parts.line.setAttribute("stroke", colour);
        parts.growingLine.setAttribute("stroke-dashoffset", parts.totalLength - Math.max(0, grownLength - headInset));
        if (parts.endHead) {
            const tip = pointAlong(element.points, grownLength);
            placeArrowHead(parts.endHead, tip.x, tip.y, tip.angle, parts.headLength);
            parts.endHead.setAttribute("fill", colour);
            parts.endHead.style.display = grownLength > 2 ? "" : "none";
        }
        if (parts.startHead) {
            const start = pointAlong(element.points, 0);
            placeArrowHead(parts.startHead, element.points[0][0], element.points[0][1], start.angle + Math.PI,
                           parts.headLength);
            parts.startHead.setAttribute("fill", colour);
        }
        if (parts.label) {
            parts.label.setAttribute("fill", cssColour(mixColour(resolveColour(element.label_colour || "dim"),
                                                                 COLOURS.accent, state.highlight)));
            parts.label.style.display = state.appearance.growth >= 1 ? "" : "none";
        }

        // Dots flowing along the arrow while it is highlighted
        parts.flowDots.forEach((dot, index) => {
            const phase = ((time * FLOW_PIXELS_PER_SECOND / parts.totalLength) + index / parts.flowDots.length) % 1;
            const position = pointAlong(element.points, phase * parts.totalLength);
            dot.setAttribute("cx", position.x);
            dot.setAttribute("cy", position.y);
            dot.setAttribute("opacity", state.highlight * Math.sin(Math.PI * phase));
        });
        return parts.flowDots.length > 0 && state.highlight > 0;
    }

    /* Text: a text at its position and anchor, or a line of segments lighting up one at a time (the tokens of a run
     * name), every segment with its colour, highlight underline and dimming. */
    function buildText(element, group) {
        const fontClass = element.font || (element.segments ? "diagram_monospace" : "diagram_text");
        if (!element.segments) {
            return { text: createText(group, element.text, element.position[0], element.position[1],
                                      element.anchor || "la", fontClass, element.colour || "text"),
                     colour: resolveColour(element.colour || "text"), segments: null };
        }
        const textElement = createSvgElement("text", { class: fontClass, x: element.position[0],
                                                       y: element.position[1], "dominant-baseline": "alphabetic" },
                                             group);
        const segments = element.segments.map((segment) => {
            const span = createSvgElement("tspan", {}, textElement);
            span.textContent = segment.text;
            return { segment: segment, span: span, underline: createSvgElement("line", {
                "stroke-width": 3, stroke: cssColour(COLOURS.accent), opacity: 0 }, group) };
        });
        return { text: textElement, colour: resolveColour(element.colour || "text"), segments: segments };
    }

    /* Text at a moment: its colour turning to the accent while highlighted, or every segment lit, underlined or
     * dimmed during its own sentences. */
    function updateText(parts, element, state, time) {
        if (!parts.segments) {
            parts.text.setAttribute("fill", cssColour(mixColour(parts.colour, COLOURS.accent, state.highlight)));
            return false;
        }
        let moving = false;
        parts.segments.forEach((record) => {
            const highlight = sentenceStateValue(record.segment.highlight, time, 0, 1, HIGHLIGHT_SECONDS);
            const dimmed = sentenceStateValue(record.segment.dim, time, 1, DIMMED_OPACITY, HIGHLIGHT_SECONDS);
            const base = record.segment.colour ? resolveColour(record.segment.colour) : parts.colour;
            record.span.setAttribute("fill", cssColour(mixColour(base, COLOURS.accent, highlight.value)));
            record.span.setAttribute("fill-opacity", dimmed.value);
            const start = record.span.getStartPositionOfChar(0).x;
            const extent = record.span.getComputedTextLength();
            record.underline.setAttribute("x1", start);
            record.underline.setAttribute("x2", start + extent);
            record.underline.setAttribute("y1", element.position[1] + 8);
            record.underline.setAttribute("y2", element.position[1] + 8);
            record.underline.setAttribute("opacity", highlight.value);
            moving = moving || highlight.moving || dimmed.moving;
        });
        return moving;
    }

    /* Image: a real image fitted into its box with its proportions (cropped first when asked, pixel scaling for small
     * images), with an optional border and caption. */
    function buildImage(element, group, imageSizes) {
        const [left, top, right, bottom] = element.box;
        const naturalSize = imageSizes[element.path];
        const crop = element.crop || [0, 0, naturalSize.width, naturalSize.height];
        const cropWidth = crop[2] - crop[0];
        const cropHeight = crop[3] - crop[1];
        const scale = Math.min((right - left) / cropWidth, (bottom - top) / cropHeight);
        const shownWidth = cropWidth * scale;
        const shownHeight = cropHeight * scale;
        const shownLeft = (left + right) / 2 - shownWidth / 2;
        const shownTop = (top + bottom) / 2 - shownHeight / 2;
        const viewport = createSvgElement("svg", { x: shownLeft, y: shownTop, width: shownWidth, height: shownHeight,
                                                   viewBox: `${crop[0]} ${crop[1]} ${cropWidth} ${cropHeight}`,
                                                   preserveAspectRatio: "none" }, group);
        const image = createSvgElement("image", { width: naturalSize.width, height: naturalSize.height,
                                                  href: element.path }, viewport);
        if (element.scaling === "pixel") {
            image.setAttribute("style", "image-rendering: pixelated");
        }
        if (element.border) {
            createSvgElement("rect", { x: shownLeft - 2, y: shownTop - 2, width: shownWidth + 4,
                                       height: shownHeight + 4, fill: "none",
                                       stroke: cssColour(resolveColour(element.border)), "stroke-width": 2 }, group);
        }
        if (element.caption) {
            createText(group, element.caption, (left + right) / 2, shownTop + shownHeight + 14, "ma",
                       element.caption_font || "diagram_small", element.caption_colour || "text");
        }
        return { outline: createHighlightOutline(group, shownLeft, shownTop, shownLeft + shownWidth,
                                                 shownTop + shownHeight) };
    }

    /* Element at a moment whose only change is its highlight outline: images, stacks, cuboids. */
    function updateOutline(parts, element, state) {
        parts.outline.setAttribute("opacity", state.highlight);
        return false;
    }

    /* Stack: layered squares standing for the channels of a feature map, each layer in its fill and opacity (the
     * channels rescaled by squeeze-and-excitation), with an optional label under the front square. */
    function buildStack(element, group) {
        const [left, top, right, bottom] = element.box;
        const count = element.count || 6;
        const offset = element.offset === undefined ? 10 : element.offset;
        const fills = element.fills || [element.fill || "blue"];
        const layerOpacities = element.layer_opacities || [1];
        for (let layerIndex = count - 1; layerIndex >= 0; layerIndex--) {
            const shift = layerIndex * offset;
            createSvgElement("rect", { x: left + shift, y: top - shift, width: right - left, height: bottom - top,
                                       fill: cssColour(resolveColour(fills[layerIndex % fills.length])),
                                       "fill-opacity": layerOpacities[layerIndex % layerOpacities.length],
                                       stroke: cssColour(resolveColour(element.outline || "card")),
                                       "stroke-width": 2 }, group);
        }
        if (element.label) {
            createText(group, element.label, (left + right) / 2, bottom + 14, "ma", element.label_font ||
                       "diagram_small", element.label_colour || "text");
        }
        return { outline: createHighlightOutline(group, left, top - (count - 1) * offset,
                                                 right + (count - 1) * offset, bottom) };
    }

    /* Bars: vertical bars with their labels under them and their values above them, or horizontal rows with their
     * label on the left and their value after the bar, growing from a hairline baseline, their data end rounded. */
    function buildBars(element, group) {
        const [left, top, right, bottom] = element.box;
        const horizontal = element.orientation === "horizontal";
        const gap = element.gap === undefined ? 6 : element.gap;
        const colours = element.colours || [element.colour || "blue"];
        const fontClass = element.font || "diagram_small";
        const count = element.values.length;
        const parts = { bars: [], valueLabels: [], horizontal: horizontal, gap: gap, fontClass: fontClass,
                        maximum: element.maximum || Math.max(...element.values),
                        labelWidth: element.label_width || 170,
                        valueWidth: element.value_labels ? (element.value_width || 100) : 0 };
        parts.rowHeight = (bottom - top - gap * (count - 1)) / count;
        parts.barWidth = (right - left - gap * (count - 1)) / count;
        element.values.forEach((value, index) => {
            parts.bars.push(createSvgElement("path", { fill: cssColour(resolveColour(colours[index % colours.length])) },
                                             group));
            if (element.labels && horizontal) {
                createText(group, element.labels[index], left + parts.labelWidth - 12,
                           top + index * (parts.rowHeight + gap) + parts.rowHeight / 2, "rm", fontClass, "text");
            } else if (element.labels) {
                createText(group, element.labels[index], left + index * (parts.barWidth + gap) + parts.barWidth / 2,
                           bottom + 10, "ma", fontClass, "dim");
            }
            if (element.value_labels) {
                parts.valueLabels.push(createSvgElement("text", { class: fontClass,
                                                                  fill: cssColour(COLOURS[horizontal ? "dim" :
                                                                                          "text"]) }, group));
            }
        });
        const baseline = horizontal ? { x1: left + parts.labelWidth, y1: top, x2: left + parts.labelWidth, y2: bottom }
            : { x1: left, y1: bottom, x2: right, y2: bottom };
        createSvgElement("line", { ...baseline, stroke: cssColour(COLOURS.border), "stroke-width": 1 }, group);
        parts.outline = createHighlightOutline(group, left, top, right, bottom);
        return parts;
    }

    /* Path of a bar with its data end rounded: horizontal from its left, or vertical from its baseline. */
    function barPath(left, top, length, thickness, horizontal) {
        if (length < 1) {
            return "";
        }
        const radius = Math.min(4, thickness / 2, length / 2);
        if (horizontal) {
            return `M ${left} ${top} h ${length - radius} a ${radius} ${radius} 0 0 1 ${radius} ${radius} ` +
                `v ${thickness - 2 * radius} a ${radius} ${radius} 0 0 1 ${-radius} ${radius} h ${-(length - radius)} z`;
        }
        return `M ${left} ${top} v ${-(length - radius)} a ${radius} ${radius} 0 0 1 ${radius} ${-radius} ` +
            `h ${thickness - 2 * radius} a ${radius} ${radius} 0 0 1 ${radius} ${radius} v ${length - radius} z`;
    }

    /* Bars at a moment: every bar as long as its value over the maximum times the growth, the values shown once grown. */
    function updateBars(parts, element, state) {
        const [left, top, right, bottom] = element.box;
        element.values.forEach((value, index) => {
            const fraction = value / parts.maximum * state.appearance.growth;
            if (parts.horizontal) {
                const thickness = Math.min(parts.rowHeight, 24);
                const centre = top + index * (parts.rowHeight + parts.gap) + parts.rowHeight / 2;
                const barLeft = left + parts.labelWidth;
                const length = (right - barLeft - parts.valueWidth) * fraction;
                parts.bars[index].setAttribute("d", barPath(barLeft, centre - thickness / 2, length, thickness, true));
                if (parts.valueLabels[index]) {
                    setMultilineText(parts.valueLabels[index], element.value_labels[index], barLeft + length + 10,
                                     centre, "lm", parts.fontClass);
                }
            } else {
                const barLeft = left + index * (parts.barWidth + parts.gap);
                const height = (bottom - top) * fraction;
                parts.bars[index].setAttribute("d", barPath(barLeft, bottom, height, parts.barWidth, false));
                if (parts.valueLabels[index]) {
                    setMultilineText(parts.valueLabels[index], element.value_labels[index],
                                     barLeft + parts.barWidth / 2, bottom - height - 8, "md", parts.fontClass);
                }
            }
            if (parts.valueLabels[index]) {
                parts.valueLabels[index].style.display = state.appearance.growth >= 1 ? "" : "none";
            }
        });
        parts.outline.setAttribute("opacity", state.highlight);
        return false;
    }

    /* Grouped bars: for every group (a class, an iteration) one bar per series (masks and boxes, trained and averaged
     * weights) side by side, every series in its colour with its name in a legend above the plot, the group names
     * along the baseline, and optionally the value of every bar. */
    function buildGroupedBars(element, group) {
        const [left, top, right, bottom] = element.box;
        const horizontal = element.orientation === "horizontal";
        const fontClass = element.font || "diagram_small";
        const groupCount = element.groups.length;
        const seriesCount = element.series.length;
        const groupGap = element.group_gap === undefined ? 24 : element.group_gap;
        const barGap = element.bar_gap === undefined ? 2 : element.bar_gap;
        const legendHeight = element.legend === false ? 0 : LINE_HEIGHTS[fontClass] + 12;
        const labelWidth = horizontal ? (element.label_width || 170) : 0;
        const plotTop = top + legendHeight;
        const parts = { bars: [], valueLabels: [], horizontal: horizontal, fontClass: fontClass, plotTop: plotTop,
                        labelWidth: labelWidth, valueWidth: horizontal && element.series.some((series) =>
                            series.value_labels) ? (element.value_width || 100) : 0,
                        maximum: element.maximum || Math.max(...element.series.flatMap((series) => series.values)) };

        // Room of every group along the group axis and of every bar within it
        const groupAxisLength = horizontal ? bottom - plotTop : right - left;
        parts.groupLength = (groupAxisLength - groupGap * (groupCount - 1)) / groupCount;
        parts.barThickness = (parts.groupLength - barGap * (seriesCount - 1)) / seriesCount;
        parts.groupGap = groupGap;
        parts.barGap = barGap;

        // One bar and one value label per series and group, the group names along the baseline
        element.series.forEach((series) => {
            const colour = cssColour(resolveColour(series.colour || "blue"));
            parts.bars.push(series.values.map(() => createSvgElement("path", { fill: colour }, group)));
            parts.valueLabels.push(series.values.map(() => series.value_labels ?
                createSvgElement("text", { class: fontClass, fill: cssColour(COLOURS.text) }, group) : null));
        });
        element.groups.forEach((name, groupIndex) => {
            const groupStart = groupIndex * (parts.groupLength + groupGap);
            if (horizontal) {
                createText(group, name, left + labelWidth - 12, plotTop + groupStart + parts.groupLength / 2, "rm",
                           fontClass, "text");
            } else {
                createText(group, name, left + groupStart + parts.groupLength / 2, bottom + 10, "ma", fontClass,
                           "dim");
            }
        });
        const baseline = horizontal ? { x1: left + labelWidth, y1: plotTop, x2: left + labelWidth, y2: bottom }
            : { x1: left, y1: bottom, x2: right, y2: bottom };
        createSvgElement("line", { ...baseline, stroke: cssColour(COLOURS.border), "stroke-width": 1 }, group);

        // Legend above the plot, a swatch and the name of every series from the right edge leftwards
        if (element.legend !== false) {
            let legendRight = right;
            [...element.series].reverse().forEach((series) => {
                const name = createText(group, series.name, legendRight, top + LINE_HEIGHTS[fontClass] / 2, "rm",
                                        fontClass, "text");
                const nameWidth = name.getComputedTextLength();
                createSvgElement("rect", { x: legendRight - nameWidth - 30, y: top + LINE_HEIGHTS[fontClass] / 2 - 9,
                                           width: 18, height: 18, rx: 4,
                                           fill: cssColour(resolveColour(series.colour || "blue")) }, group);
                legendRight -= nameWidth + 30 + 28;
            });
        }
        parts.outline = createHighlightOutline(group, left, top, right, bottom);
        return parts;
    }

    /* Grouped bars at a moment: every bar as long as its value over the maximum times the growth, the values shown
     * once grown. */
    function updateGroupedBars(parts, element, state) {
        const [left, , right, bottom] = element.box;
        element.series.forEach((series, seriesIndex) => {
            series.values.forEach((value, groupIndex) => {
                const fraction = value / parts.maximum * state.appearance.growth;
                const offset = groupIndex * (parts.groupLength + parts.groupGap) +
                    seriesIndex * (parts.barThickness + parts.barGap);
                const bar = parts.bars[seriesIndex][groupIndex];
                const label = parts.valueLabels[seriesIndex][groupIndex];
                if (parts.horizontal) {
                    const barLeft = left + parts.labelWidth;
                    const length = (right - barLeft - parts.valueWidth) * fraction;
                    const barTop = parts.plotTop + offset;
                    bar.setAttribute("d", barPath(barLeft, barTop, length, parts.barThickness, true));
                    if (label) {
                        setMultilineText(label, series.value_labels[groupIndex], barLeft + length + 10,
                                         barTop + parts.barThickness / 2, "lm", parts.fontClass);
                    }
                } else {
                    const barLeft = left + offset;
                    const height = (bottom - parts.plotTop) * fraction;
                    bar.setAttribute("d", barPath(barLeft, bottom, height, parts.barThickness, false));
                    if (label) {
                        setMultilineText(label, series.value_labels[groupIndex], barLeft + parts.barThickness / 2,
                                         bottom - height - 8, "md", parts.fontClass);
                    }
                }
                if (label) {
                    label.style.display = state.appearance.growth >= 1 ? "" : "none";
                }
            });
        });
        parts.outline.setAttribute("opacity", state.highlight);
        return false;
    }

    /* Cuboid: a tensor drawn as a box seen from above and from its right side, its front face standing for its
     * spatial size and its depth for its channels, its shape written under it. */
    function buildCuboid(element, group) {
        const [left, top, right, bottom] = element.box;
        const depth = element.depth === undefined ? 30 : element.depth;
        const shiftX = depth * Math.cos(Math.PI / 6);
        const shiftY = depth * Math.sin(Math.PI / 6);
        const fill = resolveColour(element.fill || "blue");
        const faces = [
            createSvgElement("polygon", { points: formatPoints([[left, top], [left + shiftX, top - shiftY],
                [right + shiftX, top - shiftY], [right, top]]), fill: cssColour(mixColour(fill, COLOURS.white, 0.25)) },
                             group),
            createSvgElement("polygon", { points: formatPoints([[right, top], [right + shiftX, top - shiftY],
                [right + shiftX, bottom - shiftY], [right, bottom]]),
                                          fill: cssColour(mixColour(fill, COLOURS.black, 0.35)) }, group),
            createSvgElement("rect", { x: left, y: top, width: right - left, height: bottom - top,
                                       fill: cssColour(fill) }, group)];
        faces.forEach((face) => {
            face.setAttribute("stroke", cssColour(COLOURS.card));
            face.setAttribute("stroke-width", 1.5);
            face.setAttribute("stroke-linejoin", "round");
        });
        if (element.text) {
            createText(group, element.text, (left + right) / 2, (top + bottom) / 2, "mm", element.font ||
                       "diagram_small", element.text_colour || "text");
        }
        if (element.label) {
            createText(group, element.label, (left + right + shiftX) / 2, bottom + 14, "ma",
                       element.label_font || "diagram_monospace_small", element.label_colour || "text");
        }
        return { outline: createHighlightOutline(group, left, top - shiftY, right + shiftX, bottom) };
    }

    /* Grid: a matrix of cells (similarities, costs, a confusion matrix, a heatmap), every cell coloured by its value
     * between two colours, its value written when asked, the rows and columns labelled, the cells filled in row by
     * row as it grows, and the cells of its marks outlined during their sentences. */
    function buildGrid(element, group) {
        const [left, top, right, bottom] = element.box;
        const rows = element.values.length;
        const columns = element.values[0].length;
        const cellWidth = (right - left) / columns;
        const cellHeight = (bottom - top) / rows;
        const flatValues = element.values.flat();
        const minimum = element.minimum === undefined ? Math.min(...flatValues) : element.minimum;
        const maximum = element.maximum === undefined ? Math.max(...flatValues) : element.maximum;
        const lowColour = resolveColour(element.low_colour || "box");
        const highColour = resolveColour(element.high_colour || "blue");
        const fontClass = element.font || "diagram_monospace_tiny";
        const cells = [];
        element.values.forEach((row, rowIndex) => {
            row.forEach((value, columnIndex) => {
                const mix = clampUnit(maximum > minimum ? (value - minimum) / (maximum - minimum) : 0);
                const cellGroup = createSvgElement("g", {}, group);
                createSvgElement("rect", { x: left + columnIndex * cellWidth + 1, y: top + rowIndex * cellHeight + 1,
                                           width: cellWidth - 2, height: cellHeight - 2,
                                           rx: Math.min(4, cellWidth / 6),
                                           fill: cssColour(mixColour(lowColour, highColour, mix)) }, cellGroup);
                if (element.show_values) {
                    const formatted = element.value_format === "integer" ? String(Math.round(value)) :
                        Number(value).toFixed(element.decimals === undefined ? 2 : element.decimals);
                    createText(cellGroup, formatted, left + (columnIndex + 0.5) * cellWidth,
                               top + (rowIndex + 0.5) * cellHeight, "mm", fontClass, mix > 0.55 ? "text" : "dim");
                }
                cells.push({ group: cellGroup, order: rowIndex * columns + columnIndex });
            });
        });

        // Labels of the rows on the left and of the columns above the grid
        const labelFont = element.label_font || "diagram_small";
        (element.row_labels || []).forEach((label, rowIndex) => {
            createText(group, label, left - 12, top + (rowIndex + 0.5) * cellHeight, "rm", labelFont, "dim");
        });
        (element.column_labels || []).forEach((label, columnIndex) => {
            createText(group, label, left + (columnIndex + 0.5) * cellWidth, top - 10, "md", labelFont, "dim");
        });

        // Marks: cells outlined during their sentences, such as the pairs of a Hungarian assignment
        const marks = (element.marks || []).map((mark) => ({ mark: mark, outline: createSvgElement("rect", {
            x: left + mark.column * cellWidth - 2, y: top + mark.row * cellHeight - 2, width: cellWidth + 4,
            height: cellHeight + 4, rx: 6, fill: "none", stroke: cssColour(resolveColour(mark.colour || "accent")),
            "stroke-width": 4, opacity: 0 }, group) }));
        return { cells: cells, marks: marks, outline: createHighlightOutline(group, left, top, right, bottom) };
    }

    /* Grid at a moment: the cells shown up to its growth, the marks of the active sentence outlined. */
    function updateGrid(parts, element, state, time) {
        const shownCells = state.appearance.growth * parts.cells.length;
        parts.cells.forEach((record) => {
            record.group.setAttribute("opacity", clampUnit(shownCells - record.order));
        });
        let moving = false;
        parts.marks.forEach((record) => {
            const shown = sentenceStateValue(record.mark.sentences, time, 0, 1, HIGHLIGHT_SECONDS);
            record.outline.setAttribute("opacity", shown.value);
            moving = moving || shown.moving;
        });
        parts.outline.setAttribute("opacity", state.highlight);
        return moving;
    }

    /* Curve: a line chart of one or more series in a box with its axes, ticks and labels, the series drawn from left
     * to right as it grows, the name of every series at its end. */
    function buildCurve(element, group) {
        const [left, top, right, bottom] = element.box;
        const allPoints = element.series.flatMap((series) => series.points);
        const xRange = element.x_range || [Math.min(...allPoints.map((point) => point[0])),
                                           Math.max(...allPoints.map((point) => point[0]))];
        const yRange = element.y_range || [Math.min(...allPoints.map((point) => point[1])),
                                           Math.max(...allPoints.map((point) => point[1]))];
        createSvgElement("line", { x1: left, y1: bottom, x2: right, y2: bottom, stroke: cssColour(COLOURS.border),
                                   "stroke-width": 1.5 }, group);
        createSvgElement("line", { x1: left, y1: top, x2: left, y2: bottom, stroke: cssColour(COLOURS.border),
                                   "stroke-width": 1.5 }, group);
        (element.y_ticks || []).forEach((tick) => {
            const tickY = scaleValue(tick.value, yRange[0], yRange[1], bottom, top);
            createText(group, tick.label, left - 10, tickY, "rm", "diagram_tiny", "dim");
            createSvgElement("line", { x1: left, y1: tickY, x2: right, y2: tickY, stroke: cssColour(COLOURS.border),
                                       "stroke-width": 0.75, "stroke-dasharray": "4 6" }, group);
        });
        (element.x_ticks || []).forEach((tick) => {
            createText(group, tick.label, scaleValue(tick.value, xRange[0], xRange[1], left, right), bottom + 8, "ma",
                       "diagram_tiny", "dim");
        });
        if (element.x_label) {
            createText(group, element.x_label, (left + right) / 2, bottom + 36, "ma", "diagram_small", "dim");
        }
        if (element.y_label) {
            const label = createText(group, element.y_label, left - 70, (top + bottom) / 2, "mm", "diagram_small",
                                     "dim");
            label.setAttribute("transform", `rotate(-90 ${left - 70} ${(top + bottom) / 2})`);
        }

        // Series clipped by a rectangle widening with the growth, so that they are drawn from left to right
        const clipIdentifier = `curve-clip-${engine.records.length}-${Math.round(left)}-${Math.round(top)}`;
        const clipPath = createSvgElement("clipPath", { id: clipIdentifier }, group);
        const clipRectangle = createSvgElement("rect", { x: left - 4, y: top - 40, width: 0, height: bottom - top + 80 },
                                               clipPath);
        const series = element.series.map((seriesElement) => {
            const colour = seriesElement.colour || "blue";
            const polyline = createSvgElement("polyline", {
                points: formatPoints(seriesElement.points.map((point) => [
                    scaleValue(point[0], xRange[0], xRange[1], left, right),
                    scaleValue(point[1], yRange[0], yRange[1], bottom, top)])),
                fill: "none", stroke: cssColour(resolveColour(colour)), "stroke-width": seriesElement.width || 3,
                "stroke-linejoin": "round", "stroke-linecap": "round", "clip-path": `url(#${clipIdentifier})`
            }, group);
            let name = null;
            if (seriesElement.name) {
                const last = seriesElement.points[seriesElement.points.length - 1];
                name = createText(group, seriesElement.name, scaleValue(last[0], xRange[0], xRange[1], left, right) + 10,
                                  scaleValue(last[1], yRange[0], yRange[1], bottom, top), "lm", "diagram_small",
                                  colour);
            }
            return { element: seriesElement, polyline: polyline, name: name };
        });
        separateCurveNames(series.filter((record) => record.name).map((record) => record.name), top, bottom);
        return { clipRectangle: clipRectangle, series: series };
    }

    /* Moves the names written at the last points of the series of a curve apart vertically, so that two series ending
     * at close values keep readable names: the names are sorted by height, pushed down to one line apart, then pushed
     * back up together when the lowest one leaves the plot. */
    function separateCurveNames(names, top, bottom) {
        const spacing = LINE_HEIGHTS.diagram_small;
        const firstLineHeight = (name) => Number(name.querySelector("tspan").getAttribute("y"));
        const placed = names.map((name) => ({ name: name, y: firstLineHeight(name) }))
            .sort((first, second) => first.y - second.y);
        for (let index = 1; index < placed.length; index++) {
            placed[index].y = Math.max(placed[index].y, placed[index - 1].y + spacing);
        }
        if (placed.length > 0) {
            const overflow = placed[placed.length - 1].y - bottom;
            if (overflow > 0) {
                placed.forEach((record) => { record.y = Math.max(top, record.y - overflow); });
            }
        }

        // Every name moved to its new height, the lines of a multiline name following its first line
        placed.forEach((record) => {
            const shift = record.y - firstLineHeight(record.name);
            record.name.querySelectorAll("tspan").forEach((line) => {
                line.setAttribute("y", Number(line.getAttribute("y")) + shift);
            });
        });
    }

    /* Curve at a moment: drawn up to its growth, a highlighted series thicker and a dimmed one faded. */
    function updateCurve(parts, element, state, time) {
        const [left, , right] = element.box;
        parts.clipRectangle.setAttribute("width", (right - left) * state.appearance.growth + 8);
        let moving = false;
        parts.series.forEach((record) => {
            const highlight = sentenceStateValue(record.element.highlight, time, 0, 1, HIGHLIGHT_SECONDS);
            const dimmed = sentenceStateValue(record.element.dim, time, 1, DIMMED_OPACITY, HIGHLIGHT_SECONDS);
            record.polyline.setAttribute("stroke-width", (record.element.width || 3) + highlight.value * 2);
            record.polyline.setAttribute("opacity", dimmed.value);
            if (record.name) {
                record.name.setAttribute("opacity", state.appearance.growth >= 1 ? dimmed.value : 0);
            }
            moving = moving || highlight.moving || dimmed.moving;
        });
        return moving;
    }

    /* Listing: lines of a text file read by the builder (a folder tree, a summary panel, a manifest), in a monospace
     * font, the branches of a tree dimmed, the folders in the key colour, every line highlighted by a band during its
     * sentences. */
    function buildListing(element, group) {
        const [left, top] = element.box;
        const right = element.box[2];
        const fontClass = element.font || "diagram_monospace_small";
        const lineHeight = element.line_height || LINE_HEIGHTS[fontClass] || 27;
        const lines = (element.lines || []).map((line, lineIndex) => {
            const lineTop = top + lineIndex * lineHeight;
            const band = createSvgElement("rect", { x: left, y: lineTop, width: right - left, height: lineHeight,
                                                    fill: cssColour(mixColour(COLOURS.card, COLOURS.accent, 0.18)),
                                                    opacity: line.highlight ? 1 : 0 }, group);
            const mark = createSvgElement("rect", { x: left, y: lineTop, width: 6, height: lineHeight,
                                                    fill: cssColour(COLOURS.accent), opacity: line.highlight ? 1 : 0 },
                                          group);
            const text = createSvgElement("text", { class: fontClass, x: left + 16, y: lineTop + lineHeight / 2,
                                                    "dominant-baseline": "central" }, group);
            const branchMatch = /^([│├└─\s]*)(.*)$/.exec(line.text);
            const branchSpan = createSvgElement("tspan", { fill: cssColour(COLOURS.dim) }, text);
            branchSpan.textContent = branchMatch[1].replace(/ /g, "\u00a0");
            const nameSpan = createSvgElement("tspan", { fill: cssColour(branchMatch[2].endsWith("/") ? COLOURS.key :
                                                                          COLOURS.text) }, text);
            // Spaces kept as non-breaking spaces, the same width in the monospace font, so that the columns and
            // borders of a printed panel stay aligned instead of collapsing
            nameSpan.textContent = branchMatch[2].replace(/ /g, "\u00a0");
            return { line: line, band: band, mark: mark };
        });
        return { lines: lines };
    }

    /* Listing at a moment: the bands of the lines highlighted during the active sentence. */
    function updateListing(parts, element, state, time) {
        let moving = false;
        parts.lines.forEach((record) => {
            const shown = record.line.highlight ? { value: 1, moving: false } :
                sentenceStateValue(record.line.highlight_sentences, time, 0, 1, HIGHLIGHT_SECONDS);
            record.band.setAttribute("opacity", shown.value);
            record.mark.setAttribute("opacity", shown.value);
            moving = moving || shown.moving;
        });
        return moving;
    }

    const ELEMENT_TYPES = {
        box: { build: buildBox, update: updateBox }, arrow: { build: buildArrow, update: updateArrow },
        text: { build: buildText, update: updateText }, image: { build: buildImage, update: updateOutline },
        stack: { build: buildStack, update: updateOutline }, bars: { build: buildBars, update: updateBars },
        grouped_bars: { build: buildGroupedBars, update: updateGroupedBars },
        cuboid: { build: buildCuboid, update: updateOutline }, grid: { build: buildGrid, update: updateGrid },
        curve: { build: buildCurve, update: updateCurve }, listing: { build: buildListing, update: updateListing }
    };

    /* Loads one image and resolves with its natural size. */
    function loadImageSize(path) {
        return new Promise((resolve, reject) => {
            const image = new Image();
            image.onload = () => resolve({ path: path, width: image.naturalWidth, height: image.naturalHeight });
            image.onerror = () => reject(new Error(`Cannot load the image ${path}`));
            image.src = path;
        });
    }

    /* Natural size of every image of the scene, loaded before the diagram is built. */
    async function loadImageSizes(elements) {
        const paths = [...new Set(elements.filter((element) => element.type === "image")
            .map((element) => element.path))];
        const sizes = {};
        (await Promise.all(paths.map(loadImageSize))).forEach((size) => {
            sizes[size.path] = { width: size.width, height: size.height };
        });
        return sizes;
    }

    /* Builds the scene once: the card, then every element in the order of the storyboard (later ones on top), then the
     * custom parts of the page. */
    async function setupScene(scene, timing) {
        await document.fonts.ready;
        engine.scene = scene;
        engine.timing = timing;
        engine.root = document.getElementById("scene");
        while (engine.root.firstChild) {
            engine.root.removeChild(engine.root.firstChild);
        }
        if (scene.card !== false) {
            createSvgElement("rect", { x: 1, y: 1, width: SCENE_WIDTH - 2, height: SCENE_HEIGHT - 2, rx: 18,
                                       fill: cssColour(COLOURS.card), stroke: cssColour(COLOURS.border),
                                       "stroke-width": 2 }, engine.root);
        }
        const elements = scene.elements || [];
        const imageSizes = await loadImageSizes(elements);
        engine.records = [];
        elements.forEach((element) => {
            if (!(element.type in ELEMENT_TYPES)) {
                throw new Error(`Unknown element type ${element.type}, expected one of ` +
                                `${Object.keys(ELEMENT_TYPES).join(", ")}`);
            }
            const group = createSvgElement("g", {}, engine.root);
            engine.records.push({ element: element, group: group,
                                  parts: ELEMENT_TYPES[element.type].build(element, group, imageSizes) });
        });
        engine.customUpdates = [];
        for (const customSetup of engine.customSetups) {
            engine.customUpdates.push(await customSetup({ root: engine.root, scene: scene, timing: timing,
                                                          helpers: HELPERS }));
        }
        return { elements: engine.records.length, customParts: engine.customUpdates.length };
    }

    /* Draws the scene at a time in seconds from its start; returns whether something still moves at this time. */
    function renderFrame(time) {
        let moving = false;
        engine.records.forEach((record) => {
            const appearance = computeAppearance(record.element, time);
            const dimmed = sentenceStateValue(record.element.dim, time, 1, DIMMED_OPACITY, HIGHLIGHT_SECONDS);
            const highlight = sentenceStateValue(record.element.highlight, time, 0, 1, HIGHLIGHT_SECONDS);
            record.group.setAttribute("opacity", (appearance.opacity * dimmed.value).toFixed(4));
            record.group.style.display = appearance.opacity <= 0.001 ? "none" : "";
            const state = { appearance: appearance, highlight: highlight.value };
            const elementMoving = ELEMENT_TYPES[record.element.type].update(record.parts, record.element, state, time);
            moving = moving || appearance.moving || dimmed.moving || highlight.moving || Boolean(elementMoving);
        });
        engine.customUpdates.forEach((customUpdate) => {
            moving = Boolean(customUpdate(time)) || moving;
        });
        return { moving: moving, sentence: activeSentence(time) };
    }

    /* Opacity of an SVG node as shown: the product of its own opacity and of its ancestors', 0 when one is hidden. */
    function shownOpacity(node) {
        let opacity = 1;
        for (let current = node; current && current !== engine.root; current = current.parentNode) {
            if (current.style && current.style.display === "none") {
                return 0;
            }
            const ownOpacity = current.getAttribute && current.getAttribute("opacity");
            opacity *= ownOpacity === null || ownOpacity === undefined ? 1 : Number(ownOpacity);
        }
        return opacity;
    }

    /* Layout check of the previews at a time of the scene: the shown texts reaching out of the card, and the pairs of
     * shown texts overlapping, each with the start of its text. */
    function checkLayout(time) {
        renderFrame(time);
        const shownTexts = [...engine.root.querySelectorAll("text")].filter((text) => shownOpacity(text) > 0.5)
            .map((text) => ({ text: text.textContent.slice(0, 40), rectangle: text.getBoundingClientRect() }))
            .filter((record) => record.rectangle.width > 0 && record.text.trim().length > 0);
        const problems = [];
        shownTexts.forEach((record) => {
            const rectangle = record.rectangle;
            if (rectangle.left < 4 || rectangle.top < 4 || rectangle.right > SCENE_WIDTH - 4 ||
                rectangle.bottom > SCENE_HEIGHT - 4) {
                problems.push(`text out of the card: "${record.text}"`);
            }
        });
        shownTexts.forEach((first, firstIndex) => {
            shownTexts.slice(firstIndex + 1).forEach((second) => {
                const width = Math.min(first.rectangle.right, second.rectangle.right) -
                    Math.max(first.rectangle.left, second.rectangle.left);
                const height = Math.min(first.rectangle.bottom, second.rectangle.bottom) -
                    Math.max(first.rectangle.top, second.rectangle.top);
                if (width > 3 && height > 3) {
                    problems.push(`texts overlapping: "${first.text}" and "${second.text}"`);
                }
            });
        });
        return problems;
    }

    const HELPERS = {
        COLOURS: COLOURS, resolveColour: resolveColour, cssColour: cssColour, mixColour: mixColour, ease: ease,
        clampUnit: clampUnit, scaleValue: scaleValue, formatPoints: formatPoints, createSvgElement: createSvgElement,
        createText: createText, setMultilineText: setMultilineText, sentenceStart: sentenceStart,
        activeSentence: activeSentence, sentenceStateValue: sentenceStateValue, computeAppearance: computeAppearance,
        pointAlong: pointAlong, polylineLength: polylineLength, placeArrowHead: placeArrowHead,
        APPEAR_SECONDS: APPEAR_SECONDS, GROW_SECONDS: GROW_SECONDS, HIGHLIGHT_SECONDS: HIGHLIGHT_SECONDS,
        DIMMED_OPACITY: DIMMED_OPACITY
    };

    window.Tutorial = {
        setupScene: setupScene,
        renderFrame: renderFrame,
        checkLayout: checkLayout,
        registerCustom: (customSetup) => engine.customSetups.push(customSetup),
        helpers: HELPERS
    };
})();
