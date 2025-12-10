import React from 'react';
import { useDrag } from 'react-dnd';

interface CheckerProps {
    color: number; // 1 for White (Player), -1 for Black (CPU)
    count: number;
    pointIndex: number | string; // 0-23, or 'bar'
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

    const fillColor = color > 0 ? "white" : "#d9534f"; // White vs Red
    const strokeColor = color > 0 ? "black" : "#800000";
    const textColor = color > 0 ? "black" : "white";

    return (
        <div
            ref={drag as any}
            style={{
                opacity: isDragging ? 0.5 : 1,
                cursor: canDrag ? 'grab' : 'default',
                width: '46px',
                height: '46px',
                borderRadius: '50%',
                backgroundColor: fillColor,
                border: `3px solid ${strokeColor}`,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontWeight: 'bold',
                fontSize: '0.9em',
                color: textColor,
                userSelect: 'none',
                position: 'relative',
                zIndex: 10
            }}
        >
            {count > 1 ? count : ""}
        </div>
    );
};
