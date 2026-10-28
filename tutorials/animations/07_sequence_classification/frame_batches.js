/*
 * Animated batches of frames of the sequence tutorials (sequence classification and sequence comparison): the frames of
 * real sequences moving from one layout to the next while the narration explains how the data pipeline and the frame
 * encoder handle them, from the samples of the loader to the padded batch and its mask, the gathering of the real
 * frames for the backbone, their frame vectors and their scattering back to the frame positions.
 *
 * The layout comes from a JSON file written by the figure scripts (tutorials/figures/sequence_classification and
 * sequence_comparison), named by the 'layout' of the scene data: its 'phases' in their order, its 'tiles' (a frame
 * image, an optional second image such as the vector of the frame, a fill and an outline) with their box and opacity at
 * the phases where they change, and its 'texts' with the phases they are shown in. The 'phases' of the scene data give
 * the sentence starting every phase used by the scene, so that one layout serves several scenes. A tile keeps the
 * state of the latest phase it defines, moves to its next state over MOVE_SECONDS after the start of a phase (each
 * tile delayed by the 'delay' of its state), and cross-fades to its second image when its state has 'face' 1.
 */
(function () {
    "use strict";

    // Duration of a move between two phases, and of the fade of a text, in seconds of the video
    const MOVE_SECONDS = 1.2;
    const TEXT_FADE_SECONDS = 0.45;

    /* Loads an image and resolves once it is decoded, so that the first captured frame already shows it. */
    async function preloadImage(url) {
        const image = new Image();
        image.src = url;
        await image.decode();
        return url;
    }

    /* State of a tile at a phase of the layout: the latest state it defines at or before that phase, or null. */
    function resolveTileState(tile, phaseOrder, phaseName) {
        let resolved = null;
        for (const name of phaseOrder) {
            if (tile.states[name]) {
                resolved = tile.states[name];
            }
            if (name === phaseName) {
                break;
            }
        }
        return resolved;
    }

    /* Box, opacity and face between two states at a progress of the move, a missing state being the other one
     * hidden. */
    function interpolateState(previousState, currentState, progress, ease) {
        const start = previousState || { ...currentState, opacity: 0 };
        const end = currentState || { ...previousState, opacity: 0 };
        const mix = ease(progress);
        const box = start.box.map((value, index) => value + (end.box[index] - value) * mix);
        return { box: box, opacity: start.opacity + (end.opacity - start.opacity) * mix,
                 face: (start.face || 0) + ((end.face || 0) - (start.face || 0)) * mix };
    }

    Tutorial.registerCustom(async function setupFrameBatches(context) {
        const helpers = context.helpers;
        const sceneData = context.scene.data;
        const layoutUrl = sceneData.layout;
        const layout = await (await fetch(layoutUrl)).json();

        // Phases of the scene in their order, each started by its sentence (-1 for the start of the scene)
        const scenePhases = sceneData.phases.map((phase) => ({ name: phase.name,
                                                              start: helpers.sentenceStart(phase.sentence) }));
        const unknownPhases = scenePhases.filter((phase) => !layout.phases.includes(phase.name));
        if (unknownPhases.length > 0) {
            throw new Error(`Unknown phases ${unknownPhases.map((phase) => phase.name)} of ${layoutUrl}, expected ` +
                            `some of ${layout.phases}`);
        }

        // Every image of the tiles decoded before the first frame
        const imageUrls = new Set();
        layout.tiles.forEach((tile) => {
            ["image", "second_image"].forEach((key) => {
                if (tile[key]) {
                    imageUrls.add(new URL(tile[key], layoutUrl).href);
                }
            });
        });
        await Promise.all([...imageUrls].map(preloadImage));

        // Tiles above the elements of the diagram, every tile a group with its fill, images and outline
        const layer = helpers.createSvgElement("g", {}, context.root);
        const tileRecords = layout.tiles.map((tile) => {
            const group = helpers.createSvgElement("g", {}, layer);
            const parts = { tile: tile, group: group };
            parts.fill = helpers.createSvgElement("rect", { fill: helpers.cssColour(helpers.resolveColour(
                tile.fill || "box")), rx: 3 }, group);
            if (tile.image) {
                parts.image = helpers.createSvgElement("image", { href: new URL(tile.image, layoutUrl).href,
                                                                  preserveAspectRatio: "none" }, group);
            }
            if (tile.second_image) {
                parts.secondImage = helpers.createSvgElement("image", {
                    href: new URL(tile.second_image, layoutUrl).href, preserveAspectRatio: "none", opacity: 0 },
                                                             group);
            }
            parts.outline = helpers.createSvgElement("rect", { fill: "none", rx: 3, "stroke-width": 2,
                stroke: helpers.cssColour(helpers.resolveColour(tile.outline || "border")) }, group);
            if (tile.label) {
                parts.label = helpers.createText(group, tile.label, 0, 0, "mm", "diagram_tiny", tile.label_colour ||
                                                 "dim");
            }
            return parts;
        });

        // Texts of the layout, each shown during its phases
        const textRecords = layout.texts.map((text) => ({
            text: text,
            element: helpers.createText(layer, text.text, text.position[0], text.position[1], text.anchor || "la",
                                        text.font || "diagram_small", text.colour || "text")
        }));

        /* Index of the phase of the scene running at a time, -1 before the first one. */
        function activePhaseIndex(time) {
            let active = -1;
            scenePhases.forEach((phase, index) => {
                if (time >= phase.start) {
                    active = index;
                }
            });
            return active;
        }

        /* Draws the tiles and texts at a time of the scene; returns whether something still moves. */
        return function updateFrameBatches(time) {
            const phaseIndex = Math.max(0, activePhaseIndex(time));
            const phase = scenePhases[phaseIndex];
            const previousPhase = phaseIndex > 0 ? scenePhases[phaseIndex - 1] : null;
            let moving = false;

            // Every tile between its state of the previous phase and its state of the current phase
            tileRecords.forEach((record) => {
                const currentState = resolveTileState(record.tile, layout.phases, phase.name);
                const previousState = previousPhase ? resolveTileState(record.tile, layout.phases, previousPhase.name)
                    : currentState;
                if (!currentState && !previousState) {
                    record.group.style.display = "none";
                    return;
                }
                const delay = currentState && currentState.delay ? currentState.delay : 0;
                const progress = previousPhase ? (time - phase.start - delay) / MOVE_SECONDS : 1;
                moving = moving || (previousPhase !== null && progress < 1 && time >= phase.start);
                const state = interpolateState(previousState, currentState, helpers.clampUnit(progress),
                                               helpers.ease);
                const [left, top, width, height] = state.box;
                record.group.style.display = state.opacity <= 0.001 ? "none" : "";
                record.group.setAttribute("opacity", state.opacity.toFixed(4));
                [record.fill, record.outline].forEach((rectangle) => {
                    rectangle.setAttribute("x", left);
                    rectangle.setAttribute("y", top);
                    rectangle.setAttribute("width", width);
                    rectangle.setAttribute("height", height);
                });
                [record.image, record.secondImage].forEach((image) => {
                    if (image) {
                        image.setAttribute("x", left + 2);
                        image.setAttribute("y", top + 2);
                        image.setAttribute("width", Math.max(0, width - 4));
                        image.setAttribute("height", Math.max(0, height - 4));
                    }
                });
                if (record.secondImage) {
                    record.secondImage.setAttribute("opacity", state.face.toFixed(4));
                    if (record.image) {
                        record.image.setAttribute("opacity", (1 - state.face).toFixed(4));
                    }
                }
                if (record.label) {
                    helpers.setMultilineText(record.label, record.tile.label, left + width / 2, top + height / 2,
                                             "mm", "diagram_tiny");
                    record.label.setAttribute("opacity", (1 - state.face).toFixed(4));
                }
            });

            // Every text faded in when one of its phases starts and out when it ends
            textRecords.forEach((record) => {
                const shown = record.text.phases.includes(phase.name);
                const shownBefore = previousPhase ? record.text.phases.includes(previousPhase.name) : shown;
                const fade = helpers.clampUnit((time - phase.start) / TEXT_FADE_SECONDS);
                const opacity = shown === shownBefore ? (shown ? 1 : 0) : (shown ? fade : 1 - fade);
                moving = moving || (shown !== shownBefore && fade < 1);
                record.element.setAttribute("opacity", opacity.toFixed(4));
                record.element.style.display = opacity <= 0.001 ? "none" : "";
            });
            return moving;
        };
    });
})();
