import React from 'react';

type InfoItem = {
  label: string;
  value: React.ReactNode;
};

type DynamicInfoAreaProps = {
  title: string;
  items: InfoItem[];
  className?: string;
};

const DynamicInfoArea: React.FC<DynamicInfoAreaProps> = ({
  title,
  items,
  className = ''
}) => (
  <div className={`area h-full w-full gap-[1rem] ${className}`}>
    <h3 className='subtitle ml-[0.5rem]'>{title}</h3>
    <div className='dataSum'>
      {items.map((item, idx) => (
        <div className='info' key={idx}>
          <h1>{item.label}</h1>
          <h2>{item.value}</h2>
        </div>
      ))}
    </div>
  </div>
);

export default DynamicInfoArea;
