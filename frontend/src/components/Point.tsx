import React from 'react';
import { useDrop } from 'react-dnd';
import { Checker } from './Checker';

interface PointProps {
    index: number;
    checkers: number; // + for Player, - for CPU
    onDropChecker: (fromIndex: number, toIndex: number) => void;
    legalMoves: number[][]; // Full list of [from, to]
}

export const Point: React.FC<PointProps> = ({ index, checkers, onDropChecker, legalMoves }) => {
    // Check if this point is a valid destination for ANY move is helpful for static highlighting?
    // But react-dnd `canDrop` is better.
    // Also, we can use `monitor.canDrop()` for highlighting "Valid for dragged item".

    const [{ isOver, canDrop }, drop] = useDrop(() => ({
        accept: 'CHECKER',
        canDrop: (item: { pointIndex: number, color: number }) => {
            // Must be Player's turn/color (handled by Checker drag source theoretically, but double check)
            // Validate move exists in legalMoves
            return legalMoves.some(m => m[0] === item.pointIndex && m[1] === index);
        },
        drop: (item: { pointIndex: number }) => onDropChecker(item.pointIndex, index),
        collect: (monitor) => ({
            isOver: !!monitor.isOver(),
            canDrop: !!monitor.canDrop(),
        }),
    }), [index, onDropChecker, legalMoves]); // Re-run hook if legalMoves changes

    // Visuals
    const isTop = index >= 12;
    const pointColor = index % 2 === 0 ? (isTop ? "#d2b48c" : "#8b4513") : (isTop ? "#8b4513" : "#d2b48c");

    // Checkers list
    const count = Math.abs(checkers);
    const checkerColor = checkers > 0 ? 1 : -1;

    // User Request: If > 5 pieces, show 5 and number on top
    const MAX_VISUAL = 5;
    const renderCount = Math.min(count, MAX_VISUAL);

    const checkerParams = [];
    for (let i = 0; i < renderCount; i++) {
        // If this is the LAST visible checker (top of stack) AND there are more...
        // We pass the TOTAL count to be displayed
        const isTopStack = (i === renderCount - 1) && (count > MAX_VISUAL);
        checkerParams.push({
            countDisplay: isTopStack ? count : 1,
            isTopStack
        });
    }

    // Highlight: Yellow if over and valid, LightGreen if dragging and valid target
    // canDrop means "Some item is being dragged and it CAN be dropped here".
    // isOver means "The item is currently hovering over this".
    const highlight = isOver && canDrop ? 'yellow' : (canDrop ? 'rgba(0, 255, 0, 0.2)' : 'transparent');

    return (
        <div
            ref={drop as any} // Explicit cast for DnD v15/React18 compat
            style={{
                flex: 1,
                height: '100%',
                backgroundColor: highlight,
                position: 'relative',
                display: 'flex',
                flexDirection: isTop ? 'column' : 'column-reverse',
                alignItems: 'center',
                padding: '5px 0'
            }}
        >
            {/* Triangle Graphic */}
            <div style={{
                position: 'absolute',
                top: 0, bottom: 0, left: 0, right: 0,
                zIndex: 0,
                backgroundColor: pointColor,
                clipPath: isTop ? 'polygon(50% 100%, 0 0, 100% 0)' : 'polygon(50% 0, 0 100%, 100% 100%)',
                opacity: 0.8
            }} />

            {/* Checkers */}
            {checkerParams.map((param: any, i) => (
                <Checker
                    key={i}
                    color={checkerColor}
                    count={param.countDisplay}
                    pointIndex={index}
                    canDrag={checkerColor > 0}
                />
            ))}

            {/* Label */}
            <div style={{ position: 'absolute', [isTop ? 'top' : 'bottom']: 0, fontSize: '10px', zIndex: 20, color: isTop ? 'white' : 'black', fontWeight: 'bold' }}>
                {index}
            </div>
        </div>
    );
}
