import React from 'react';
import Styles from './SideBar.module.css';

type SidebarButtonProps = {
  icon: string; // FontAwesome icon class, e.g. 'fa-solid fa-house'
  text: string;
  active?: boolean;
  disabled?: boolean;
  onClick?: () => void;
};

const SidebarButton: React.FC<SidebarButtonProps> = ({
  icon,
  text,
  active = false,
  disabled = false,
  onClick
}) => (
  <button
    className={active ? `${Styles.tabBut} ${Styles.active}` : Styles.tabBut}
    disabled={disabled}
    onClick={onClick}
    type='button'
  >
    <span className={Styles.iconWrapper}>
      <i className={icon}></i>
    </span>
    <span className={Styles.tabText}>{text}</span>
  </button>
);

export default SidebarButton;
