import React from 'react';
import { useDrag } from 'react-dnd';

interface CheckerProps {
    color: number; // 1 for White (Player), -1 for Black (CPU)
    count: number;
    pointIndex: number; // 0-23, or special ID
    canDrag: boolean;
}

export const Checker: React.FC<CheckerProps> = ({ color, count, pointIndex, canDrag }) => {
    const [{ isDragging }, drag] = useDrag(() => ({
        type: 'CHECKER',
        item: { color, pointIndex },
        canDrag: () => canDrag,
        collect: (monitor) => ({
            isDragging: !!monitor.isDragging(),
        }),
    }), [canDrag, pointIndex, color]);

    const fillColor = color > 0 ? "white" : "#d9534f";
    const strokeColor = color > 0 ? "black" : "#800000";
    const textColor = color > 0 ? "black" : "white";

    return (
        <div
            ref={drag}
            style={{
                opacity: isDragging ? 0.5 : 1,
                cursor: canDrag ? 'grab' : 'default',
                width: '40px',
                height: '40px',
                borderRadius: '50%',
                backgroundColor: fillColor,
                border: `2px solid ${strokeColor}`,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontWeight: 'bold',
                color: textColor,
                userSelect: 'none',
                position: 'relative',
                marginBottom: '-30px', // Stack effect
                zIndex: 10
            }}
        >
            {count > 1 ? count : ""}
        </div>
    );
};
