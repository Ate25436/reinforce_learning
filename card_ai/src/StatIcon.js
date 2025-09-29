function StatIcon({ icon, value, x, y }) {
    return (
        <>
            <image href={icon} x={x} y={y} width="30" height="30" />
            <text
                x={x + 15}
                y={y + 22}
                textAnchor="middle"
                fill="white"
                stroke="black"
                strokeWidth="1"
                fontSize="20"
                fontWeight="bold"
            >
                {value}
            </text>
        </>
    );
}
export default StatIcon;
