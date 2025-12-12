import React from 'react';
import { useDragLayer, XYCoord } from 'react-dnd';
import { Checker } from './Checker';

const layerStyles: React.CSSProperties = {
    position: 'fixed',
    pointerEvents: 'none',
    zIndex: 100,
    left: 0,
    top: 0,
    width: '100%',
    height: '100%',
};

function getItemStyles(initialOffset: XYCoord | null, currentOffset: XYCoord | null) {
    if (!initialOffset || !currentOffset) {
        return {
            display: 'none',
        };
    }

    let { x, y } = currentOffset;

    const transform = `translate(${x}px, ${y}px)`;
    return {
        transform,
        WebkitTransform: transform,
    };
}

export const CustomDragLayer: React.FC = () => {
    const {
        itemType,
        isDragging,
        item,
        initialOffset,
        currentOffset,
    } = useDragLayer((monitor) => ({
        item: monitor.getItem(),
        itemType: monitor.getItemType(),
        initialOffset: monitor.getInitialSourceClientOffset(),
        currentOffset: monitor.getSourceClientOffset(),
        isDragging: monitor.isDragging(),
    }));

    if (!isDragging || !item) {
        return null;
    }

    function renderItem() {
        switch (itemType) {
            case 'CHECKER':
                // Render a version of the Checker purely for display (no drag handlers)
                // We pass canDrag=false just to satisfy props, but it's visual only.
                return <Checker color={item.color} count={1} pointIndex="preview" canDrag={false} isPreview={true} />;
            default:
                return null;
        }
    }

    return (
        <div style={layerStyles}>
            <div style={getItemStyles(initialOffset, currentOffset)}>
                {renderItem()}
            </div>
        </div>
    );
};
