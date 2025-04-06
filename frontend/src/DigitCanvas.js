import React, { useRef, useState, useEffect, forwardRef, useImperativeHandle } from 'react';

const DigitCanvas = forwardRef((props, ref) => {
    const canvasRef = useRef(null);
    const [isDrawing, setIsDrawing] = useState(false);
    const [context, setContext] = useState(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        ctx.strokeStyle = 'black'; // Drawing color
        ctx.lineWidth = 15;        // Line thickness - adjust for better recognition
        ctx.lineCap = 'round';     // Smoother lines
        ctx.fillStyle = 'white';   // Background color
        ctx.fillRect(0, 0, canvas.width, canvas.height); // Fill background
        setContext(ctx);
    }, []);

    const startDrawing = ({ nativeEvent }) => {
        const { offsetX, offsetY } = nativeEvent;
        context.beginPath();
        context.moveTo(offsetX, offsetY);
        setIsDrawing(true);
    };

    const draw = ({ nativeEvent }) => {
        if (!isDrawing) return;
        const { offsetX, offsetY } = nativeEvent;
        context.lineTo(offsetX, offsetY);
        context.stroke();
    };

    const stopDrawing = () => {
        context.closePath();
        setIsDrawing(false);
    };

    const clearCanvas = () => {
        if (context) {
            const canvas = canvasRef.current;
            context.fillStyle = 'white'; // Ensure background is white on clear
            context.fillRect(0, 0, canvas.width, canvas.height);
        }
    };

    // Expose methods to parent component using ref
    useImperativeHandle(ref, () => ({
        clearCanvas,
        getImageDataUrl: () => {
            if (canvasRef.current) {
                // Get the Data URL (base64 encoded image)
                return canvasRef.current.toDataURL('image/png');
            }
            return null;
        }
    }));


    return (
        <canvas
            ref={canvasRef}
            width={280} // Use larger canvas for easier drawing, will be downscaled
            height={280}
            style={{ border: '1px solid black', touchAction: 'none' }} // touchAction prevents scrolling on mobile
            onMouseDown={startDrawing}
            onMouseMove={draw}
            onMouseUp={stopDrawing}
            onMouseLeave={stopDrawing} // Stop drawing if mouse leaves canvas
            // Add touch events for mobile support
            onTouchStart={startDrawing}
            onTouchMove={draw}
            onTouchEnd={stopDrawing}
        />
    );
});

export default DigitCanvas;
