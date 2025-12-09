import React from 'react';
import { useDrop } from 'react-dnd';
import { Checker } from './Checker';

interface PointProps {
    index: number;
    checkers: number; // + for Player, - for CPU
    onDropChecker: (fromIndex: number, toIndex: number) => void;
    canMoveTo: boolean; // Highlight legal move
}

export const Point: React.FC<PointProps> = ({ index, checkers, onDropChecker, canMoveTo }) => {
    const [{ isOver }, drop] = useDrop(() => ({
        accept: 'CHECKER',
        drop: (item: { pointIndex: number }) => onDropChecker(item.pointIndex, index),
        collect: (monitor) => ({
            isOver: !!monitor.isOver(),
        }),
    }), [index, onDropChecker]);

    // Visuals
    const isTop = index >= 12;
    const pointColor = index % 2 === 0 ? (isTop ? "#d2b48c" : "#8b4513") : (isTop ? "#8b4513" : "#d2b48c");

    // Checkers list
    const count = Math.abs(checkers);
    const checkerColor = checkers > 0 ? 1 : -1;

    const checkerParams = [];
    for (let i = 0; i < count; i++) {
        checkerParams.push(i);
    }

    return (
        <div
            ref={drop as any} // Explicit cast for DnD v15/React18 compat
            style={{
                flex: 1,
                height: '100%',
                backgroundColor: isOver ? 'yellow' : (canMoveTo ? 'lightgreen' : 'transparent'),
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
            {checkerParams.map((_, i) => (
                <Checker
                    key={i}
                    color={checkerColor}
                    count={i === count - 1 ? count : 1}
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
